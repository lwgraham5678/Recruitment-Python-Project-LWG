import Testdefs as td
from TestConfig import P
import statistics as s

for n in range(1,100):
    dst = td.GetDegreeSequences(P.points)

    dsc = td.oddtest(dst[0])
    dsx = td.oddtest(dst[1])


    GL = td.CreateNetworks(dsc, dsx)
    P.overlapstat = td.FindOverlapStat(GL[0], GL[1])

    P.overlapstatlist.append(P.overlapstat)

print(s.mean(P.overlapstatlist))