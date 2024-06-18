import numpy as np
from Global_Vars import Global_Vars
from Score import reliefF


def objfun(Soln):
    data = Global_Vars.Feat
    Tar = Global_Vars.Target
    Fitn = np.zeros(Soln.shape[0])
    dimension = len(Soln.shape)
    if dimension == 2:
        for i in range(Soln.shape[0]):
            sol = Soln[i, :]
            halfLen = round(sol.shape[0] // 2)
            feat = sol[:halfLen].astype(np.int16)
            weight = sol[halfLen:]
            data1 = data[:, feat] * weight
            rscore = reliefF(np.array(data1), np.array(Tar.reshape(-1)))
            Fitn[i] = 1 / rscore
        return Fitn
    else:
        sol = Soln
        halfLen = round(sol.shape[0] // 2)
        feat = sol[:halfLen].astype(np.int16)
        weight = sol[halfLen:]
        data1 = data[:, feat] * weight
        rscore = reliefF(np.array(data1), np.array(Tar.reshape(-1)))
        Fitn = 1 / rscore
        return Fitn
