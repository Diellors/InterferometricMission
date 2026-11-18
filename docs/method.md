# Modeli dhe metoda

## Dinamika

Përdorim sistemin rrotullues të problemit rrethor të kufizuar me tre trupa.
Në njësitë e normalizuara, trupat kryesorë janë në (-mu,0,0) dhe (1-mu,0,0).
Për pozicionin r dhe shpejtësinë v, përshpejtimi ka përbërësit centrifugale
(x,y,0), Coriolis (2vy,-2vx,0) dhe tërheqjen gravitacionale të dy trupave.

Për një trup me masë m dhe zhvendosje d, gradienti i përshpejtimit është:
m * (3 d d^T / |d|^5 - I / |d|^3).
Shtojmë gradientin centrifugal dhe pjesët Coriolis në Jacobianin A.

Matrica e tranzicionit plotëson dPhi/dt = A(t) Phi, Phi(0) = I.
Integrimi përfshin 6 komponentë të orbitës dhe 36 komponentë të Phi.
RK4 adaptiv krahason një hap të plotë me dy gjysmëhapa; diferenca /15
vlerëson gabimin dhe përdoret për korrigjim Richardson.
Toleranca lokale e shkallëzuar është 1e-10. Ajo nuk garanton të njëjtin
gabim global, sidomos pranë drejtimeve të paqëndrueshme HALO.

Pozicioni relativ në një matje është pjesa pozicionale e Phi(t) delta_s(0).
Faktori fizik i përbashkët i shkallëzimit anulohet në këtë model linear.
Kushtet fillestare dhe periudhat janë parametra të detyrës.

## Objektivi

Pas përjashtimit të pikave jashtë kubit, koordinatat kuantizohen me
int(G*p/2 + G//2). int shkurton drejt zeros.
Për çdo plan, set-i i diferencave përfshin orientimet pozitive dhe negative.
Diferenca zero numërohet vetëm kur ka të paktën një pikë.
Mbivendosjet dhe diferencat e përsëritura numërohen vetëm një herë.

Minimizojmë negativin e matjes më të dobët. Ky është interpretimi i
përdorur në projekt, pavarësisht mospërputhjes max/min të përshkrimit të
mëparshëm të sfidës. Nuk kemi verifikuar rezultatet kundrejt portalit.

## Kërkimi

Inicializimi përdor gjeometrinë e kufizimeve në të tria matjet.
Matrica A përmban pjesën pozicionale të matricave të tranzicionit, të
normalizuara me faktorin e shtrirjes, dhe kufijtë e variablave.
Faktorizimi Cholesky i A^T A përdoret për të reskaluar drejtimet e kërkimit:
drejtimet e paqëndrueshme zvogëlohen, kurse drejtimet e qëndrueshme mbeten të gjera.

Algoritmi `feasible-satellite-search-v2` ndryshon vetëm një satelit për provë.
Kombinon lëvizje përgjatë segmenteve që respektojnë kufizimet, zëvendësime me
gjendje të reja, eksplorim pranë kufijve dhe kombinim të gjendjeve nga individë
të ndryshëm. Kandidatët mund të nisen nga individi aktual ose nga më i miri.

Objektivi kryesor mbetet negativi i matjes më të dobët. Vetëm kur ky objektiv
është i barabartë, përdoret shuma e tri matjeve për të zgjedhur kandidatin.
Vlerësimi inkremental përditëson numërimet e baselines vetëm për satelitin që
lëviz. Testet e krahasojnë me llogaritjen e plotë, përfshirë mbivendosjet dhe
pikat që hyjnë ose dalin nga rrjeti. Normalizimi nuk ka ndryshuar.

`--resume` nis një kërkim të ri nga një rezultat i verifikuar dhe disa mutacione
pranë tij; nuk rikthen gjendjen e plotë të popullatës ose RNG-së së ekzekutimit
të mëparshëm. Buxheti i vlerësimeve i përket vetëm ekzekutimit të ri.

## Kufiri i rastit Easy

Me N satelitë, ka maksimumi N(N-1) diferenca të orientuara jo-zero dhe një zero.
Për N=5 dhe G=11, secili plan ka maksimumi 21/(21^2) = 1/21 fill factor.
Shuma e tri planeve ka maksimumi 1/7, prandaj fitness >= -1/7 = -0.142857142857.
Një rezultat që e arrin këtë kufi është optimal për këtë model dhe normalizim.
Vlera -1.5 është e pamundur për Easy; nuk ndryshojmë shkallën për ta imituar.
