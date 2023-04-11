
def f(x, alpha, beta, gamma, delta):
    if x < alpha:
        return gamma
    if x > beta:
        return delta
    if alpha <= x <= beta:
        return (delta-gamma)/(beta-alpha)*x+(gamma*beta-delta*alpha)/(beta-alpha)