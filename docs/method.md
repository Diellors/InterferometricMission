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

Inicializimi reskalon çdo gjendje të rastësishme që pozicionet relative
të qëndrojnë brenda rrjetit në të tria matjet.
DE/rand/1/bin kombinon tre individë të ndryshëm nga objektivi.
Crossover detyron të paktën një komponent nga mutanti.
Kufijtë zbatohen me prerje dhe ruajmë kandidatin kur nuk përkeqëson fitness-in.
Përzgjedhja është sinkrone, prandaj i gjithë brezi lexon popullatën e vjetër.

Ky është algoritëm fillestar. Në HALO, reskalimi mund të grumbullojë
pikat pranë qendrës dhe të kufizojë cilësinë e kërkimit.
Krahaso disa seeds dhe buxhete të barabarta vlerësimesh para përfundimeve.
