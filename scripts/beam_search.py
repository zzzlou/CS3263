import evaluate
def search(state_spaces, beam_width, permutation_dict):
    beam = [[]]
    for domain in state_spaces:
        new_beam = []
        for state in beam:
            for value in domain:
                new_state = state + [value]
                new_beam.append(new_state)
        new_beam.sort(key=lambda state: evaluate.f(state, permutation_dict), reverse=True)
        beam = new_beam[:beam_width]
    best_state = max(beam, key=lambda state: evaluate.f(state, permutation_dict))
    return best_state
