# Algoritmi lakmitar me përmirësim lokal

Ky variant është shkruar për projektin dhe synon të jetë i shpjegueshëm në
lëndën Dizajn dhe Analizë e Algoritmeve. Ai kombinon dy teknika të njohura:
ndërtim lakmitar (greedy) dhe kërkim lokal me përmirësimin e parë.
Nuk pretendojmë risi shkencore ose garanci të optimumit global.

## Hyrja dhe dalja

- N: numri i satelitëve që duhen zgjedhur.
- C: një listë me K >= N gjendje orbitale të realizueshme, secila me 6 komponentë.
- M: numri i matjeve, 3 në këtë projekt.
- G: madhësia e rrjetit.
- E: buxheti i vlerësimeve të kandidatëve.
- S: numri maksimal i kalimeve të kërkimit lokal.

Dalja është një përzgjedhje prej N indekseve të ndryshme të C, vektori x dhe
fitness-i i rivlerësuar. Gjendjet në listë mund të projektohen në të njëjtat
qeliza; indekset e përzgjedhura mbeten të ndryshme.

Lista krijohet nga kampionuesi orbital i projektit me një seed të caktuar.
Pas krijimit të listës, vetë kërkimi është determinist: barazimet ruajnë
indeksin e parë. Ndryshimi i seed-it ndryshon listën, jo rregullin greedy.

## Kriteri i zgjedhjes

Për matjen t, F_t është shuma e fill factors në XY, XZ dhe YZ.
Krahasojmë çelësin leksikografik:

    key(x) = (-min_t F_t(x), -sum_t F_t(x))

Komponenti i parë është saktësisht objektivi ekzistues. I dyti shërben vetëm
për barazime. Objektivi dhe normalizimi nuk ndryshojnë.
Gjatë ndërtimit, F_t llogaritet për formacionin e pjesshëm, me të njëjtin
emërues të rrjetit; ai nuk raportohet si rezultat final me N satelitë.

## Pseudokodi

```text
Satelitët = bosh
Baselines = bosh për çdo matje dhe plan

për h = 1 deri N:
    më_i_miri = asnjë
    për çdo kandidat c që nuk është përzgjedhur:
        numëro baselines e reja midis c dhe satelitëve të përzgjedhur
        vlerëso key(Satelitët + c)
        ruaj c nëse key është më i vogël se kandidati më i mirë
    shto kandidatin më të mirë dhe përditëso Baselines

përsërit deri S kalime:
    u_përmirësua = jo
    për çdo pozicion i të formacionit:
        për çdo kandidat c jashtë përzgjedhjes:
            nëse buxheti ka mbaruar: kthe formacionin aktual
            vlerëso zëvendësimin e satelitit i me c
            nëse key përmirësohet rreptësisht:
                prano zëvendësimin
                u_përmirësua = po
                vazhdo te pozicioni tjetër
    nëse nuk u_përmirësua:
        kthe formacionin si optimum lokal brenda listës

rivlerëso formacionin final me funksionin e plotë të projektit
```

## Saktësia dhe përfundimi

Pas hapit h të ndërtimit kemi h indekse të ndryshme dhe bashkësitë e sakta
të baselines për këta satelitë. Kjo vlen për h=0. Shtimi i kandidatit krijon
vetëm diferencën zero dhe diferencat e orientuara me satelitët ekzistues.
Bashkimi i tyre ruan invariantin për h+1. Pra pas N hapave kemi formacion
të vlefshëm dhe numërime të sakta. Ky argument nuk provon optimalitet greedy.

Çdo zëvendësim ruan numrin N dhe indekset e ndryshme. Ai pranohet vetëm me
key rreptësisht më të vogël, prandaj fitness-i kryesor nuk përkeqësohet.
Ka një numër të fundëm përzgjedhjesh, maksimumi binomial(K,N), kështu që
përzgjedhjet me të njëjtën vlerë nuk mund të krijojnë cikël përmirësimesh.
Gjithashtu buxheti E dhe kufiri S e ndalin ekzekutimin.

`pool_local_optimum` raportohet vetëm pas një kalimi të plotë pa përmirësim:
në atë formacion u kontrolluan të gjitha zëvendësimet e një sateliti nga C.
`evaluation_limit` dhe `sweep_limit` nuk japin këtë garanci.
Optimumi lokal është vetëm kundrejt kësaj liste dhe këtij fqinjësimi;
jo ndaj të gjitha gjendjeve të vazhdueshme të problemit orbital.
Nuk po pretendojmë një kufi standard përafrimi për këtë objektiv max-min.

## Kompleksiteti

Analiza supozon kosto mesatare O(1) për operacionet hash të seteve/fjalorëve.
Dimensioni i gjendjes është konstante 6 dhe ka tri plane për matje.
Integrimi fillestar për matricat orbitale trajtohet veçmas.

- Projektimi i listës: O(MK).
- Në hapin h, një shtim kontrollon O(Mh) diferenca. Për K kandidatë dhe N
  hapa, ndërtimi ka kosto O(M K N^2).
- Le të jetë B = min(N^2, (2G-1)^2), kufi për numrin e baselines në një plan.
  Një zëvendësim kopjon numërime deri në O(MB), kontrollon O(N) koordinata
  dhe ndryshon O(MN) diferenca: O(M(N+B)). Nuk është O(1).
- Me R prova zëvendësimi, përmirësimi lokal kushton O(R M(N+B)), ku
  R <= S N(K-N) dhe R <= E - [NK - N(N-1)/2].
- Hapësira: O(M(K+N+B) + E), duke përfshirë listën, matricat e qelizave,
  numërimet dhe gjurmët e ekzekutimit. Mbetet e kufizuar nga buxheti i dhënë.

## Ekzekutimi dhe matja

```bash
python scripts/run_greedy.py --case easy --pool-size 80 --max-evaluations 4000 --seed 42
python scripts/run_greedy.py --case medium --pool-size 80 --max-evaluations 4000 --seed 42
python scripts/run_greedy.py --case hard --pool-size 80 --max-evaluations 4000 --seed 42
python -m unittest discover -s tests -p test_greedy.py -v
```

Rezultati JSON ruhet nën results/greedy në një folder të ri.
Verifikoje me `python scripts/verify_solution.py RRUGA_E_JSON`.
Ekzekutimet nuk mbishkruajnë rezultate ekzistuese.

`construction_evaluations` numëron formacione të pjesshme, kurse
`refinement_evaluations` formacione të plota. Shuma e tyre është `evaluations`.
Rivlerësimi final numërohet veçmas si `verification_evaluations=1`.
Këto nuk duhen barazuar pa shpjegim me buxhetin e optimizuesit tjetër,
i cili vlerëson vetëm formacione të plota.

Gjurmët ruhen në `construction_trace` dhe `improvement_trace`, sepse hapat
greedy nuk janë breza të popullatës. Grafiku ekzistues i brezave nuk paraqet
automatikisht këto gjurmë. Ky variant është alternativë e analizueshme;
duhet eksperiment i kontrolluar për të pretenduar se është më i mirë.
