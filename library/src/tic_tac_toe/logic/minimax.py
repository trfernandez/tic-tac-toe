# tic_tac_toe/logic/minimax.py

from functools import cache, partial

from tic_tac_toe.logic.models import GameState, Mark, Move

def find_best_move(game_state: GameState) -> Move | None:
	maximizer: Mark = game_state.current_mark
	bound_minimax = partial(minimax_alphabeta, maximizer=maximizer)
	return max(game_state.possible_moves, key=bound_minimax)

def minimax(move: Move, maximizer: Mark, choose_highest_score: bool = False) -> int:
	if move.after_state.game_over:
		return move.after_state.evaluate_score(maximizer)
	return (max if choose_highest_score else min) (
		minimax(next_move, maximizer, not choose_highest_score)
		for next_move in move.after_state.possible_moves)

@cache
def minimax_alphabeta(move: Move, maximizer: Mark, 
	alpha=-1, beta=1, choose_highest_score: bool = False) -> int:
	if move.after_state.game_over:
		return move.after_state.evaluate_score(maximizer)

	moves = []
	for next_move in move.after_state.possible_moves:
		moves.append(
			score := minimax_alphabeta(next_move, maximizer, alpha, beta, not choose_highest_score))
		if choose_highest_score:
			alpha = max(alpha, score)
		else:
			beta = min(beta, score)
		if beta <= alpha:
			break

	return (max if choose_highest_score else min)(moves)