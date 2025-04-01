class CSP:
    def __init__(self, variables, domains, neighbors, permu_dict):
        """
        variables: list of variable names (e.g., ['X1', 'X2', ...])
        domains: dict mapping each variable to its list of possible values.
                 For example, {'X1': [0, 1, ..., k], ...}
        neighbors: dict mapping each variable to a list of variables with which it has a binary constraint.
        constraints: a function that takes (xi, x, xj, y) and returns True if the assignment is allowed.
        """
        self.variables = variables
        self.domains = domains
        self.neighbors = neighbors
        self.event_dict = permu_dict
    
    def constraints(self, xi, x, xj, y):
        def non_overlap(ls_x, ls_y):
            for event_x in ls_x:
                for event_y in ls_y:
                    if event_x.day == event_y.day:
                        if (event_x.end_time > event_y.start_time and event_x.start_time < event_y.end_time) or (event_y.end_time > event_x.start_time and event_y.start_time < event_x.end_time):
                            return False
            return True
        ls_x = self.event_dict[xi][x]
        ls_y = self.event_dict[xj][y]
        return non_overlap(ls_x, ls_y)


def ac3(csp):
    """
    Applies the AC-3 algorithm to enforce arc consistency.
    Returns False if an inconsistency is found (a domain is emptied); otherwise, returns True.
    """
    queue = [(xi, xj) for xi in csp.variables for xj in csp.neighbors[xi]]
    while queue:
        xi, xj = queue.pop(0)
        if revise(csp, xi, xj):
            if not csp.domains[xi]:
                return False
            for xk in csp.neighbors[xi]:
                if xk != xj:
                    queue.append((xk, xi))
    return True

def revise(csp, xi, xj):
    """
    Revise the domain of xi by removing values that are inconsistent with xj.
    Returns True if any value is removed from the domain of xi.
    """
    revised = False
    for x in csp.domains[xi][:]:
        if all(not csp.constraints(xi, x, xj, y) for y in csp.domains[xj]):
            csp.domains[xi].remove(x)
            revised = True
    return revised