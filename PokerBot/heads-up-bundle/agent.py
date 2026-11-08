"""
agent.py - YOUR poker bot. This one file is a complete web server.

Run it:
    python agent.py                  # listens on port 5008
    PORT=5010 python agent.py        # or pick a port

Play a match against a baseline (three windows):
    python agent.py                                 # your bot on port 5008
    python baseline-bots/baseline_random.py         # opponent on port 5009
    python judge_engine.py                          # judge: both halves (uses those two ports)

Check that the tournament can run it:
    python -m pytest -q              # tests in test_agent.py
    ./tools/check_build.sh           # builds and smoke-tests it like the tournament

TODO:
    1. Assign your agent a unique PARTICIPANT and AGENT_NAME below.
    2. Implement the three functions between "YOUR CODE HERE" and "END OF YOUR CODE".

Everything under "SERVER PLUMBING" talks to the judge. You normally never
need to change it, and if you break it the judge can't talk to your bot.
"""

import os
import traceback

import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse

from evaluator import card_rank, card_suit, cards_to_str, omaha_score, score_five, warmup  # handy helpers, see evaluator.py
from game import Action, AgentConfig, HandResult, Observation, Street

# Build omaha_score's lookup table now, while the server starts, instead of
# during your first decision (where it would eat into your time bank).
# Do any other slow setup (loading a model, ...) here too: the server must
# answer within 15 seconds of starting.
warmup()

# rename as your team/bot's name
PARTICIPANT = "Dead Man's Hand"
AGENT_NAME = "LuckyCallingStation"


# ===== YOUR CODE HERE =====

def new_half(config: AgentConfig, memory: dict) -> None:
    """Called at the start of each half-match (1000 hands).

    `memory` is a fresh, empty dict every half. Store anything you want to
    remember between hands in it, e.g. memory["opp_raises"] = 0.
    `config` holds the match settings (starting_stack, big_blind, ...).

    In the tournament, each half runs in a brand-new copy of your server,
    so nothing you learn in one half carries over to the other half.
    """
    memory["hands_played"] = 0
    memory["opponent_actions"] = 0
    memory["opponent_raises"] = 0
    memory["opponent_calls"] = 0
    memory["last_hand_index"] = None
    memory["processed_actions"] = 0


def choose_action(obs: Observation, memory: dict) -> Action:
    """Called every time it is your turn. Return one of obs.legal_actions.

    What you can see in obs:
        hand_index          which hand of the half this is (0, 1, 2, ...)
        street              Street.PREFLOP, Street.FLOP, Street.TURN or Street.RIVER
        hole_cards          your 5 cards as ints, sorted ascending (0-35)
        board               board (community) cards seen so far (0, 3, 4 or 5 ints)
        pot                 total chips both players put in this hand
        to_call             chips you must add to match the opponent
        my_bet, opp_bet     chips you,opponent put in on THIS street
        my_stack, opp_stack chips each player still has behind
        is_button           True if you are the button (small blind)
        legal_actions       the Actions you may return right now
        raise_to            {RAISE_HALF: total, RAISE_POT: total} for legal raises
        min_raise_to        smallest normal raise total (all-ins may be lower)
        raises_this_street  raises so far this street (max 4)
        action_history      this hand's actions so far; each has
                            street, is_me, action, amount, street_total
        big_blind           size of the big blind in chips

    Actions: Action.FOLD, Action.CHECK_CALL, Action.RAISE_HALF, Action.RAISE_POT
    Cards are ints: card_rank(c) is 0 (six) .. 8 (ace), card_suit(c) is 0..3,
    and cards_to_str(obs.hole_cards) prints them like "6h 7c 9s Kd Ah".
    Once all 5 board cards are out, omaha_score(obs.hole_cards, obs.board)
    scores your hand (higher is better).
    """

    def raise_action(preferred: Action) -> Action:
        if preferred in obs.legal_actions:
            return preferred
        if Action.RAISE_HALF in obs.legal_actions:
            return Action.RAISE_HALF
        if Action.RAISE_POT in obs.legal_actions:
            return Action.RAISE_POT
        return Action.CHECK_CALL

    # Track only opponent actions seen in the current hand.  The action history
    # is cumulative, so reset its counter whenever a new hand begins.
    if memory.get("last_hand_index") != obs.hand_index:
        memory["last_hand_index"] = obs.hand_index
        memory["processed_actions"] = 0
    for action in obs.action_history[memory["processed_actions"]:]:
        if not action.is_me:
            memory["opponent_actions"] += 1
            if action.action == Action.RAISE_HALF or action.action == Action.RAISE_POT:
                memory["opponent_raises"] += 1
            elif action.action == Action.CHECK_CALL and action.amount > 0:
                memory["opponent_calls"] += 1
    memory["processed_actions"] = len(obs.action_history)

    opponent_actions = memory["opponent_actions"]
    opponent_call_rate = (
        memory["opponent_calls"] / opponent_actions if opponent_actions >= 30 else 0.5
    )
    opponent_raise_rate = (
        memory["opponent_raises"] / opponent_actions if opponent_actions >= 30 else 0.5
    )

    ranks = [card_rank(card) for card in obs.hole_cards]
    rank_counts = {rank: ranks.count(rank) for rank in set(ranks)}
    suits = [card_suit(card) for card in obs.hole_cards]
    suit_counts = {suit: suits.count(suit) for suit in set(suits)}
    unique_ranks = sorted(set(ranks))

    pair_count = sum(count // 2 for count in rank_counts.values())
    trip_count = sum(count // 3 for count in rank_counts.values())
    max_suit_count = max(suit_counts.values(), default=0)

    longest_run = 1
    run = 1
    for index in range(1, len(unique_ranks)):
        if unique_ranks[index] == unique_ranks[index - 1] + 1:
            run += 1
            longest_run = max(longest_run, run)
        else:
            run = 1
    wheel_ranks = set(unique_ranks) | {0}
    if {8, 0, 1, 2, 3}.issubset(wheel_ranks):
        longest_run = max(longest_run, 5)

    high_cards = sum(rank >= 6 for rank in ranks)
    preflop_score = (
        sum(ranks)
        + 18 * pair_count
        + 25 * trip_count
        + 3 * longest_run
        + 3 * max_suit_count
        + 3 * high_cards
    )
    if obs.is_button:
        preflop_score += 4

    # Pot odds are the amount paid divided by the total pot after calling.
    pot_after_call = obs.pot + obs.to_call
    pot_odds = obs.to_call / pot_after_call if pot_after_call else 0.0

    if obs.street == Street.PREFLOP:
        if obs.to_call == 0:
            if preflop_score >= 52:
                return raise_action(Action.RAISE_POT)
            if preflop_score >= 38 and obs.is_button:
                return raise_action(Action.RAISE_HALF)
            return Action.CHECK_CALL

        if opponent_raise_rate >= 0.75:
            if preflop_score >= 65:
                return raise_action(Action.RAISE_POT)
            if preflop_score >= 50 and pot_odds <= 0.35:
                return Action.CHECK_CALL
            return Action.FOLD

        if preflop_score >= 55:
            return raise_action(Action.RAISE_HALF)
        if preflop_score >= 42 and pot_odds <= 0.45:
            return Action.CHECK_CALL
        if preflop_score >= 34 and pot_odds <= 0.25 and opponent_call_rate >= 0.55:
            return Action.CHECK_CALL
        return Action.FOLD

    # Approximate the best Omaha five-card result with the actual cards
    # available.  On the river, this is the exact evaluator path.
    best_score = 0
    hole_pairs = [
        (first, second)
        for first in range(len(obs.hole_cards))
        for second in range(first + 1, len(obs.hole_cards))
    ]
    if len(obs.board) == 5:
        best_score = omaha_score(obs.hole_cards, obs.board)
    else:
        board_triples = [
            (first, second, third)
            for first in range(len(obs.board))
            for second in range(first + 1, len(obs.board))
            for third in range(second + 1, len(obs.board))
        ]
        for hole_first, hole_second in hole_pairs:
            for board_first, board_second, board_third in board_triples:
                five_cards = (
                    obs.hole_cards[hole_first],
                    obs.hole_cards[hole_second],
                    obs.board[board_first],
                    obs.board[board_second],
                    obs.board[board_third],
                )
                best_score = max(best_score, score_five(five_cards))

    category = best_score >> 20
    is_nuts_or_near_nuts = category >= 5
    is_made_hand = category >= 3

    if obs.to_call > 0:
        if is_nuts_or_near_nuts:
            if obs.raises_this_street < 4 and opponent_raise_rate < 0.85:
                return raise_action(Action.RAISE_POT)
            return Action.CHECK_CALL
        if opponent_raise_rate >= 0.75:
            if category >= 4 and pot_odds <= 0.50:
                return Action.CHECK_CALL
            if category == 3 and pot_odds <= 0.25:
                return Action.CHECK_CALL
            return Action.FOLD
        if is_made_hand and pot_odds <= 0.55:
            return Action.CHECK_CALL
        if category == 2 and pot_odds <= 0.30:
            return Action.CHECK_CALL
        if category == 1 and pot_odds <= 0.18 and opponent_call_rate >= 0.60:
            return Action.CHECK_CALL
        return Action.FOLD

    if is_nuts_or_near_nuts:
        return raise_action(
            Action.RAISE_POT if obs.street == Street.RIVER else Action.RAISE_HALF
        )
    if is_made_hand:
        return raise_action(Action.RAISE_HALF)
    if category == 2 and obs.is_button:
        return raise_action(Action.RAISE_HALF)
    return Action.CHECK_CALL


def on_hand_end(result: HandResult, memory: dict) -> None:
    """Called once for every finished hand. Optional: learn from what happened.

    To save time, the judge sends finished hands along with your next
    decision request, so this always runs BEFORE choose_action() sees the
    next hand. (The last hands of a half arrive at the end of the half.)

    result has: hand_index, my_net (chips won or lost), pot,
    went_to_showdown, board (cards revealed), opp_hole (opponent's cards,
    only after a showdown, else None), winner ('me', 'opp' or 'split').
    """
    memory["hands_played"] += 1

# ===== END OF YOUR CODE =====


# ===== SERVER PLUMBING (you normally don't need to edit below) =====
# The judge calls these routes:
#   GET  /           -> who you are (also used as the health check)
#   POST /new-half   -> start a fresh half: memory is wiped
#   POST /act        -> the judge sends any finished hands plus an Observation;
#                       on_hand_end() runs for each hand, then you answer an action
#   POST /hand-end   -> the judge sends the last finished hands of a half
#
# Why "async def"? The routes run on one thread and call your functions
# directly, so your functions never run twice at once. Your own functions
# stay normal "def" (don't make them async).
#
# Keep your per-half state in `memory`. In the tournament every half runs in
# a fresh server process, so everything starts over. When you test locally,
# though, one process can play both halves and only `memory` is wiped, NOT
# other global variables. Using `memory` makes your bot behave the same way
# in both places.

app = FastAPI()
MEMORY: dict = {}                                   # your memory for this half
STATUS = {"started": False}                         # has /new-half been called?
TOKEN = os.environ.get("AGENT_TOKEN") or None       # set by the tournament


async def _read_request(request: Request, path: str) -> tuple[dict, JSONResponse | None]:
    """Check the token (and that the half has started) BEFORE reading the body.

    Checking first means a request without the right token costs your bot
    almost nothing, so nobody else can slow it down by sending junk.
    Returns (body, None) when the request is fine, or ({}, error response).
    """
    if TOKEN and request.headers.get("x-agent-token") != TOKEN:
        return {}, JSONResponse({"error": "bad or missing X-Agent-Token"}, status_code=401)
    if not STATUS["started"] and path != "/new-half":
        return {}, JSONResponse({"error": "call /new-half first"}, status_code=409)
    try:
        body = await request.json()
    except ValueError:
        body = None
    if not isinstance(body, dict):
        return {}, JSONResponse({"error": "the request body must be a JSON object"}, status_code=400)
    return body, None


def _error_info(exc: Exception) -> dict:
    """Describe an exception from your code (call this inside `except`)."""
    return {"error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()}


def _crash(exc: Exception, hand_end_errors: list | None = None) -> JSONResponse:
    """Report an exception from your code to the judge (with traceback)."""
    return JSONResponse({**_error_info(exc), "hand_end_errors": hand_end_errors or []},
                        status_code=500)


def _deliver_results(raw_results) -> list:
    """Call on_hand_end() for each finished hand the judge sent.
    Returns a list of error reports (one per hand that crashed)."""
    errors = []
    for raw in raw_results or []:
        try:
            on_hand_end(HandResult.from_dict(raw), MEMORY)
        except Exception as exc:
            errors.append(_error_info(exc))
    return errors


@app.get("/")
async def info():
    """Health check: tells the judge who this bot is."""
    return {"participant": PARTICIPANT, "agent_name": AGENT_NAME}


@app.post("/new-half")
async def route_new_half(request: Request):
    """Wipe memory and call your new_half(). The body is {"config": {...}}."""
    body, refused = await _read_request(request, "/new-half")
    if refused:
        return refused
    MEMORY.clear()
    STATUS["started"] = True
    try:
        new_half(AgentConfig.from_dict(body.get("config", {})), MEMORY)
    except Exception as exc:
        return _crash(exc)
    return {"ok": True}


@app.post("/act")
async def route_act(request: Request):
    """Deliver finished hands, then turn the JSON into an Observation, call
    choose_action(), and send the answer back."""
    body, refused = await _read_request(request, "/act")
    if refused:
        return refused
    hand_end_errors = _deliver_results(body.get("hand_results"))
    try:
        obs = Observation.from_dict(body["observation"])
        action = choose_action(obs, MEMORY)
    except Exception as exc:
        return _crash(exc, hand_end_errors)
    # Send the action's name; anything strange is passed on for the judge to reject.
    if isinstance(action, Action):
        answer = action.name
    elif action is None or isinstance(action, (str, int, float, bool)):
        answer = action
    else:
        answer = repr(action)
    return {"action": answer, "hand_end_errors": hand_end_errors}


@app.post("/hand-end")
async def route_hand_end(request: Request):
    """Call on_hand_end() for the last finished hands of a half."""
    body, refused = await _read_request(request, "/hand-end")
    if refused:
        return refused
    return {"ok": True, "hand_end_errors": _deliver_results(body.get("results"))}


if __name__ == "__main__":
    # The tournament sets PORT as an environment variable, so each bot is compliant to our evaluation platform
    # Locally it defaults to 5008 for your own testing convenience
    port = int(os.environ.get("PORT", "5008"))
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info", access_log=False)
