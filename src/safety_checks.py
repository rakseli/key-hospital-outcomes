
def dominance(values, n=3, k=90):

    # Dominance: based on absolute non-zero contributions
    contribs = sorted((abs(v) for v in values if v != 0), reverse=True)
    total = sum(contribs)
    share = sum(contribs[:n]) / total if total > 0 else float("nan")
    print(total)
    print(share)
    if total > 0 and share > (k / 100):
        sensitive = True
    else:
        sensitive = False
    
    return sensitive

def min_freq(values,n=10):
    if len(values)<10:
        return True
    else:
        return False