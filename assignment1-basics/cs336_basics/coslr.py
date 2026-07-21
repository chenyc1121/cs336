import math

def coslr(t,alpha_max,alpha_min,tw,tc):
    if t<tw:
        return t*alpha_max/tw
    elif tw<=t<=tc:
        return alpha_min+0.5*(1+math.cos((t-tw)*math.pi/(tc-tw)))*(alpha_max-alpha_min)
    else :
        return alpha_min
