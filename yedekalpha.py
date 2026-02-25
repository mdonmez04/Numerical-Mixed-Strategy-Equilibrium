# /main.py
import argparse
import math

import jax
import optax
from jax import lax, nn, random
from jax import numpy as jnp
from jax.scipy.stats import rankdata
from jax.scipy.special import logsumexp
from matplotlib import pyplot as plt


# ------------------------------------------------------------
# Small helpers
# ------------------------------------------------------------

def delete(lst, index):
    return lst[:index] + lst[index:][1:]


def replace(lst, index, item):
    return lst[:index] + [item] + lst[index:][1:]


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--game", default="Alwaysone")

    # 3 players: [Merged, Independent, Type-2 Consumer]
    # Player-2 has 2 pure actions (I vs M), so sizes[2] MUST be 2.
    p.add_argument("--sizes", type=int, default=[31, 32, 2], nargs="*")

    p.add_argument("--freeze_support", type=int, default=0)
    p.add_argument("--freeze_weight", type=int, default=0)

    p.add_argument("--opt", default="sgd")
    p.add_argument("--lr", type=float, default=0.06)

    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--iters", type=int, default=17000)
    return p.parse_args()


# ------------------------------------------------------------
# Normal-form game utilities
# ------------------------------------------------------------

def create_nfg(f, supports):
    """
    Create a payoff tensor by evaluating f on all combinations of support points.

    tensor shape: (*n_actions_per_player, n_players) then moved to (n_players, *n_actions)
    """
    in_axes = [None] * len(supports)
    for i in reversed(range(len(supports))):
        f = jax.vmap(f, [replace(in_axes, i, 0)])
    tensor = f(supports)
    assert tensor.shape == (*map(len, supports), len(supports))
    return jnp.moveaxis(tensor, -1, 0)


def get_nfg_utilities(utilities, strategies):
    for strategy in reversed(strategies):
        utilities @= strategy
    return utilities


def get_nfg_player_utilities(utilities, strategies, player):
    utilities = utilities[player]
    utilities = jnp.moveaxis(utilities, player, 0)
    for strat in reversed(delete(strategies, player)):
        utilities @= strat
    return utilities


def get_nfg_exploitability(utilities, strategies):
    br_values = [
        get_nfg_player_utilities(utilities, strategies, player).max()
        for player in range(len(strategies))
    ]
    current_utilities = get_nfg_utilities(utilities, strategies)
    gaps = jnp.stack(br_values) - current_utilities
    return gaps.sum()


def mix(x, axis=-1):
    ranks = rankdata(x, method="ordinal", axis=axis)
    weights = ranks / ranks.shape[axis]
    return jnp.vecdot(x, weights, axis=axis)


# ------------------------------------------------------------
# Action spaces
# ------------------------------------------------------------

class Interval:
    def discretize(self, points):
        return jnp.linspace(0.0, 1.0, int(points))

    def reparameterize(self, x):
        return nn.sigmoid(x)

    def reparameterize_inverse(self, x):
        return jax.scipy.special.logit(x)

    def shape(self):
        return ()


class Box01:
    def __init__(self, d):
        self.d = int(d)

    def discretize(self, points):
        points = int(points)
        k = int(math.ceil(math.sqrt(points)))
        grid = jnp.linspace(0.0, 1.0, k)
        X, Y = jnp.meshgrid(grid, grid, indexing="ij")
        pts = jnp.stack([X.reshape(-1), Y.reshape(-1)], axis=-1)
        return pts[:points]

    def reparameterize(self, x):
        return nn.sigmoid(x)

    def reparameterize_inverse(self, x):
        return jax.scipy.special.logit(x)

    def shape(self):
        return (self.d,)


class Simplex:
    def __init__(self, n):
        self.n = int(n)

    def reparameterize(self, x):
        return nn.softmax(x)

    def shape(self):
        return (self.n,)


class BinaryActions01:
    """
    Two pure actions encoded as scalars:
      0.0 = choose Independent on 2nd search
      1.0 = choose Merged again on 2nd search

    IMPORTANT: These are labels, not probabilities.
    Alpha will come from the *weights* over these two actions.
    """

    def discretize(self, points):
        
        return jnp.array([0.0, 1.0])

    def reparameterize(self, x):
        # identity (support points are already the action labels)
        return x

    def reparameterize_inverse(self, x):
        # identity
        return x

    def shape(self):
        return ()


# ------------------------------------------------------------
# Game: 3 players
#   Player 0: merged chooses (pL, pH) with constraint pH>=pL (via (uL,g))
#   Player 1: independent chooses pI
#   Player 2: type-2 chooses {I, M} in the pH-information set (binary)
#
# Alpha = weight on action "I" (encoded as action=0.0)
# ------------------------------------------------------------

class AlwayspHone:
    def __init__(
        self,
        r=1.0,
        lambdas=(0.7, 0.1, 0.2),
        costs=(0.0, 0.0, 0.0),
        demand=1.0,
        tau=1e-2,
    ):
        assert 0.0 < r <= 1.0
        lam1, lam2, lam3 = lambdas
        self.r = float(r)
        self.lambdas = (float(lam1), float(lam2), float(lam3))
        self.costs = tuple(map(float, costs))   # (cL, cH, cI)
        self.D = float(demand)
        self.tau = float(tau)

    def pricing_region(self):
        lam1, lam2, lam3 = self.lambdas
        return "joint" if lam2**2 >= 3.0 * lam1 * lam3 else "distinct"

    def action_space(self, player):
        if player == 0:
            return Box01(2)            # merged picks (uL,g)
        if player == 1:
            return Interval()          # independent picks uI
        if player == 2:
            return BinaryActions01()   # type-2 picks action label {0,1}
        raise ValueError("bad player index")

    # -------------------------
    # smooth helpers
    # -------------------------

    def _soft_pair_alloc(self, x, y):
        """
        Returns (share_to_x, share_to_y) for unit mass choosing the lower price,
        using a smooth sigmoid with temperature tau.
        """
        tau = self.tau
        share_x = nn.sigmoid((y - x) / tau)  # high when x < y
        return share_x, 1.0 - share_x

    def _softmin(self, x, y):
        """
        Smooth min(x,y) via log-sum-exp:
          min(x,y) ~ -tau * log(exp(-x/tau) + exp(-y/tau))
        """
        tau = self.tau
        return -tau * logsumexp(jnp.stack([-x / tau, -y / tau], axis=-1), axis=-1)

    # -------------------------
    # Payoffs
    # -------------------------

    def utilities(self, actions):
        """
        actions are pure actions (support points) for each player:
          a0: (uL,g) -> pL=r*uL, pH=r*(uL+(1-uL)*g)
          a1: uI     -> pI=r*uI
          a2: 0.0 or 1.0
               0.0 => choose Independent on second search in the (saw pH) node
               1.0 => choose Merged again on second search in the (saw pH) node
        returns: [pi_M, pi_I, U2]
        """
        a0, a1, a2 = actions

        # merged prices
        uL = a0[..., 0]
        g = a0[..., 1]
        uH = uL + (1.0 - uL) * g
        pL = self.r * uL
        pH = self.r * uH

        # independent price
        pI = self.r * a1

        # consumer action label (pure action here)
        # define:
        #   choose_I = 1 if action==0, else 0
        #   choose_M = 1 if action==1, else 0
        choose_I = 1.0 - a2
        choose_M = a2

        lam1, lam2, lam3 = self.lambdas
        cL, cH, cI = self.costs
        D = self.D
        dtype = pL.dtype

        # -------------------------
        # Type-1: equal split
        # -------------------------
        q1 = jnp.array([lam1 / 3.0, lam1 / 3.0, lam1 / 3.0], dtype=dtype)

        # -------------------------
        # Type-2: your explicit two-search logic
        #
        # First search is random over the three prices: pL, pH, pI with prob 1/3 each.
        # He knows firm identity.
        #
        # If first saw Independent (pI): no strategic action; second search is Merged.
        #   (Assumption retained from before: the merged price revealed is pL or pH with prob 1/2 each.)
        #
        # If first saw Merged:
        #   - if first saw pL: he infers pL and will NOT search merged again; second search is Independent.
        #   - if first saw pH and pH==r in equilibrium: he is at the strategic node
        #       * if action = I: second search Independent; he can buy from I if cheaper
        #       * if action = M: second search Merged again; he cannot buy from I in this branch
        #
        # Buys after second search at the lowest observed price.
        # No search costs.
        # -------------------------

        one_third = jnp.array(1.0 / 3.0, dtype=dtype)
        one_half = jnp.array(1.0 / 2.0, dtype=dtype)

        # ----- Case A: first sees pL (merged low). Second sees pI (independent).
        # He can buy from either, min(pL,pI).
        share_L_A, share_I_A = self._soft_pair_alloc(pL, pI)
        q2_L_A = share_L_A
        q2_H_A = jnp.array(0.0, dtype=dtype)
        q2_I_A = share_I_A
        purchase_A = self._softmin(pL, pI)

        # ----- Case B: first sees pH (merged high; inferred pH node).
        # If choose I: second search independent; can buy from I if cheaper than pH.
        share_H_BI, share_I_BI = self._soft_pair_alloc(pH, pI)
        q2_L_BI = jnp.array(0.0, dtype=dtype)
        q2_H_BI = share_H_BI
        q2_I_BI = share_I_BI
        purchase_BI = self._softmin(pH, pI)

        # If choose M: second search merged again -> sees pL deterministically
        # and since he never searched I, he CANNOT buy from independent here.
        q2_L_BM = jnp.array(1.0, dtype=dtype)
        q2_H_BM = jnp.array(0.0, dtype=dtype)
        q2_I_BM = jnp.array(0.0, dtype=dtype)
        purchase_BM = pL

        # Combine by *pure action label* (choose_I or choose_M are 0/1 here)
        q2_L_B = choose_I * q2_L_BI + choose_M * q2_L_BM
        q2_H_B = choose_I * q2_H_BI + choose_M * q2_H_BM
        q2_I_B = choose_I * q2_I_BI + choose_M * q2_I_BM
        purchase_B = choose_I * purchase_BI + choose_M * purchase_BM

        # ----- Case C: first sees pI (independent). Second sees merged.
        # Assumption: merged reveals pL or pH with prob 1/2 each (since none seen yet).
        # If reveal pL: buy min(pI,pL)
        share_I_CL, share_L_CL = self._soft_pair_alloc(pI, pL)
        q2_I_CL = share_I_CL
        q2_L_CL = share_L_CL
        q2_H_CL = jnp.array(0.0, dtype=dtype)
        purchase_CL = self._softmin(pI, pL)

        # If reveal pH: buy min(pI,pH)
        share_I_CH, share_H_CH = self._soft_pair_alloc(pI, pH)
        q2_I_CH = share_I_CH
        q2_H_CH = share_H_CH
        q2_L_CH = jnp.array(0.0, dtype=dtype)
        purchase_CH = self._softmin(pI, pH)

        q2_L_C = one_half * q2_L_CL + one_half * q2_L_CH
        q2_H_C = one_half * q2_H_CL + one_half * q2_H_CH
        q2_I_C = one_half * q2_I_CL + one_half * q2_I_CH
        purchase_C = one_half * purchase_CL + one_half * purchase_CH

        # ----- Combine cases with 1/3 each, then multiply by lambda2 mass
        q2_L = lam2 * (one_third * q2_L_A + one_third * q2_L_B + one_third * q2_L_C)
        q2_H = lam2 * (one_third * q2_H_A + one_third * q2_H_B + one_third * q2_H_C)
        q2_I = lam2 * (one_third * q2_I_A + one_third * q2_I_B + one_third * q2_I_C)
        q2 = jnp.stack([q2_L, q2_H, q2_I], axis=-1)

        # Type-2 expected purchase price and utility
        purchase_2 = one_third * purchase_A + one_third * purchase_B + one_third * purchase_C
        U2 = lam2 * (self.r - purchase_2)

        # -------------------------
        # Type-3: softmax on prices
        # -------------------------
        P = jnp.stack([pL, pH, pI], axis=-1)
        alloc3 = nn.softmax(-P / self.tau, axis=-1)
        q3 = lam3 * alloc3

        # Total shares
        shares = q1 + q2 + q3
        qL, qH, qI = shares[..., 0], shares[..., 1], shares[..., 2]

        # Profits
        pi_M = D * ((pL - cL) * qL + (pH - cH) * qH)
        pi_I = D * ((pI - cI) * qI)

        return jnp.stack([pi_M, pi_I, U2], axis=-1)

    # ------------------------------------------------------------------
    # Theoretical CDFs (unchanged)
    # ------------------------------------------------------------------
    def distinct_pricing_cdf(self, player, grid):
        lam1, lam2, lam3 = self.lambdas
        r = self.r
        denom = (lam1 + 2.0 * lam2 + 3.0 * lam3)
        pD = (lam1 + lam2) * r / denom
        eps = 1e-12
        p = jnp.maximum(grid, eps)
        FD = 1.0 - ((lam1 + lam2) * (r - p)) / ((lam2 + 3.0 * lam3) * p)
        FD = jnp.where(grid < pD, 0.0, FD)
        FD = jnp.where(grid >= r, 1.0, FD)
        return jnp.clip(FD, 0.0, 1.0)

    def joint_pricing_cdf(self, player, grid):
        lam1, lam2, lam3 = self.lambdas
        r = self.r
        denom = (2.0 * lam1 + 3.0 * lam2 + 3.0 * lam3)
        pJ = (2.0 * lam1 + lam2) * r / denom
        eps = 1e-12
        p = jnp.maximum(grid, eps)
        FJ = 1.0 - ((2.0 * lam1 + lam2) * (r - p)) / ((2.0 * lam2 + 3.0 * lam3) * p)
        FJ = jnp.where(grid < pJ, 0.0, FJ)
        FJ = jnp.where(grid >= r, 1.0, FJ)
        return jnp.clip(FJ, 0.0, 1.0)

    def get_theoretical_cdf(self, player, num_points=500):
        grid = jnp.linspace(0.0, self.r, num_points)
        F = self.joint_pricing_cdf(player, grid) if self.pricing_region() == "joint" else self.distinct_pricing_cdf(player, grid)
        return grid, F

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------
    def plot_strategy(self, player, support, weight):
        if player == 0:
            uL = support[:, 0]
            g = support[:, 1]
            uH = uL + (1.0 - uL) * g
            pL = self.r * uL
            pH = self.r * uH
            self._plot_1d_cdf(pL, weight, title="Player 0 (pL)")
            self._plot_1d_cdf(pH, weight, title="Player 0 (pH)")
        elif player == 1:
            pI = self.r * support
            self._plot_1d_cdf(pI, weight, title="Player 1 (pI)")
        elif player == 2:
            # support = [0,1] where 0=I,1=M; weights = [alpha, 1-alpha]
            fig, ax = plt.subplots()
            alpha = float(weight[0])
            ax.bar(["I (alpha)", "M (1-alpha)"], [float(weight[0]), float(weight[1])])
            ax.set_ylim(0.0, 1.0)
            ax.set_title(f"Player 2 (Type-2 mixing)  alpha={alpha:.4f}")
            ax.set_ylabel("prob")
        else:
            raise ValueError("bad player index")

    def _plot_1d_cdf(self, prices, weight, title):
        order = prices.argsort()
        sorted_support = prices[order]
        learned_cdf = weight[order].cumsum()

        fig, ax = plt.subplots()
        ax.step(sorted_support, learned_cdf, where="post", label="learned CDF")

        grid, theor_cdf = self.get_theoretical_cdf(player=0)
        ax.plot(grid, theor_cdf, linestyle="--", label=f"theoretical CDF ({self.pricing_region()})")

        ax.set(title=title, xlabel="price", ylabel="cum prob")
        ax.set_xlim(0.0, self.r)
        ax.set_ylim(0.0, 1.0)
        ax.legend()


# ------------------------------------------------------------
# Support / weight utilities
# ------------------------------------------------------------

def get_support(params, game):
    return [
        jax.vmap(game.action_space(player).reparameterize)(p)
        for player, p in enumerate(params["support"])
    ]


def get_weight(params):
    return [Simplex(p.size).reparameterize(p) for p in params["weight"]]


def plot_params(params, game):
    support = get_support(params, game)
    weight = get_weight(params)
    for player, (p_support, p_weight) in enumerate(zip(support, weight)):
        game.plot_strategy(player, p_support, p_weight)


def get_exploit(params, key, game):
    support = get_support(params, game)
    weight = get_weight(params)

    nfg = create_nfg(game.utilities, support)

    def get_br_utility(player):
        new_support = replace(
            support,
            player,
            game.action_space(player).discretize(200 if player != 2 else 2),
        )
        new_nfg = create_nfg(game.utilities, new_support)
        utilities = get_nfg_player_utilities(new_nfg, weight, player)
        return utilities.max()

    num_players = len(weight)
    br_utilities = [get_br_utility(player) for player in range(num_players)]
    regrets = jnp.stack(br_utilities) - get_nfg_utilities(nfg, weight)
    return regrets.sum()


def get_metagame_exploit(params, key, game):
    support = get_support(params, game)
    weight = get_weight(params)
    nfg = create_nfg(game.utilities, support)
    return get_nfg_exploitability(nfg, weight)


def get_support_params_grads(params, key, game):
    def f(new_params, old_params):
        new_support = get_support(new_params, game)
        old_support = get_support(old_params, game)
        weight = get_weight(params)

        def g(player):
            mix_support = replace(old_support, player, new_support[player])
            nfg = create_nfg(game.utilities, mix_support)
            utilities = get_nfg_player_utilities(nfg, weight, player)
            return -mix(utilities)

        num_players = len(new_support)
        return sum(map(g, range(num_players)))

    return jax.grad(f)(params, params)["support"]


def get_weight_params_grads(params, key, game):
    return jax.grad(get_metagame_exploit)(params, key, game)["weight"]


def get_metrics(params, key, game):
    return {
        "exploitability": get_exploit(params, key, game),
        "metagame exploitability": get_metagame_exploit(params, key, game),
    }


# ------------------------------------------------------------
# Game / optimizer selection
# ------------------------------------------------------------

def get_game(args):
    match args.game:
        case "Alwaysone":
            return AlwayspHone()
        case _:
            raise NotImplementedError


def get_optimizer(args):
    match args.opt:
        case "sgd":
            return optax.sgd(args.lr)
        case _:
            raise NotImplementedError


# ------------------------------------------------------------
# Initialization
# ------------------------------------------------------------

def create_state(key, sizes, game, optimizer):
    support_params = []
    for player, size in enumerate(sizes):
        space = game.action_space(player)

        grid = space.discretize(size)

        # For Interval/Box01 we avoid exact 0/1 to keep logits finite.
        # For BinaryActions01 we keep [0,1] exactly (labels).
        if isinstance(space, (Interval, Box01)):
            eps = 1e-6
            grid = jnp.clip(grid, eps, 1.0 - eps)

        init_params = jax.vmap(space.reparameterize_inverse)(grid)
        support_params.append(init_params)

    # weights: unconstrained logits -> softmax
    weight_params = [jnp.zeros(int(size)) for size in sizes]
    params = {"support": support_params, "weight": weight_params}
    opt_state = optimizer.init(params)
    return {"params": params, "opt_state": opt_state}


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------

def main():
    args = parse_args()
    assert len(args.sizes) == 3, "sizes must be length 3: [merged, independent, consumer]"
    assert args.sizes[2] == 2, "consumer must have exactly 2 actions, so sizes[2]=2"

    game = get_game(args)
    optimizer = get_optimizer(args)

    def get_grads(params, key):
        # Support grads
        if args.freeze_support:
            support_grads = optax.tree_utils.tree_zeros_like(params["support"])
        else:
            support_grads = get_support_params_grads(params, key, game)

        # IMPORTANT: consumer support must be fixed to [0,1] (binary action labels)
        # so we always freeze player-2 support gradients.
        support_grads = list(support_grads)
        support_grads[2] = jnp.zeros_like(support_grads[2])
        # DO NOT convert to tuple — keep list to match params["support"]
        # support_grads = tuple(support_grads)  # <-- remove this

        # Weight grads
        if args.freeze_weight:
            weight_grads = optax.tree_utils.tree_zeros_like(params["weight"])
        else:
            weight_grads = get_weight_params_grads(params, key, game)

        return {"support": support_grads, "weight": weight_grads}

    def update_state(state, key):
        grads = get_grads(state["params"], key)
        updates, opt_state = optimizer.update(grads, state["opt_state"])
        params = optax.apply_updates(state["params"], updates)
        new_state = {"params": params, "opt_state": opt_state}
        return new_state, get_metrics(params, key, game)

    key = random.key(args.seed)
    key, subkey = random.split(key)
    init_state = create_state(subkey, args.sizes, game, optimizer)

    keys = random.split(key, args.iters)
    final_state, history = lax.scan(update_state, init_state, keys)

    # --- print alpha as ONE NUMBER (weight on consumer action "I")
    final_weights = get_weight(final_state["params"])
    alpha = float(final_weights[2][0])          # prob(action=I) where action=0.0
    one_minus_alpha = float(final_weights[2][1])

    print("\n=== Type-2 Mixing ===")
    print(f"alpha = P(search Independent | saw pH) = {alpha:.6f}")
    print(f"1-alpha = P(search Merged again | saw pH) = {one_minus_alpha:.6f}")

    print("\n=== Final metrics ===")
    print(f"exploitability (last) = {float(history['exploitability'][-1]):.6f}")
    print(f"metagame exploitability (last) = {float(history['metagame exploitability'][-1]):.6f}")

    # --- plots
    plt.rcParams["figure.constrained_layout.use"] = True
    plt.rcParams["savefig.dpi"] = 300

    for metric in history:
        fig, ax = plt.subplots()
        ax.plot(history[metric])
        ax.set(xlabel="iteration", ylabel=metric, title=metric)

    plot_params(final_state["params"], game)
    plt.show()


if __name__ == "__main__":
    main()