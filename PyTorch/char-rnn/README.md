The Jean de la Fontaine writings can be accessed freely online. 

The text was slightly preprocessed to remove the titles "FABLES I/II/III/..." and also some introductory texts by Jean de la Fontaine.

See also [https://github.com/kevinboone/epub2txt2](https://github.com/kevinboone/epub2txt2)

## Installation

```
python3 -m venv venv
source venv/bin/activate 
python -m pip install -r requirements.txt
```

## Running

```
source venv/bin/activate 
python main.py train --num_cells 512 --num_layers 2 --slength 30
```

A training epoch takes around 3s on a recent GPU but up to 1 min either on a decent CPU only machine.

The performances should already reach 25% accuracy at the end of the first epoch and quickly ramps up to 50% for both training and validation before the accuracy metric ramping up again to 70%. 

## Example output 

Below is a trace of the program for the first 40 epochs or so.

```
INFO:root:Loading the data
INFO:root:The conversion map is {'\t': 0, '\n': 1, ' ': 2, '!': 3, '"': 4, "'": 5, '(': 6, ')': 7, '*': 8, '+': 9, ',': 10, '-': 11, '.': 12, '/': 13, '0': 14, '1': 15, '4': 16, '5': 17, '6': 18, '8': 19, '9': 20, ':': 21, ';': 22, '<': 23, '>': 24, '?': 25, 'A': 26, 'B': 27, 'C': 28, 'D': 29, 'E': 30, 'F': 31, 'G': 32, 'H': 33, 'I': 34, 'J': 35, 'L': 36, 'M': 37, 'N': 38, 'O': 39, 'P': 40, 'Q': 41, 'R': 42, 'S': 43, 'T': 44, 'U': 45, 'V': 46, 'W': 47, 'X': 48, 'Y': 49, 'Z': 50, '[': 51, ']': 52, 'a': 53, 'b': 54, 'c': 55, 'd': 56, 'e': 57, 'f': 58, 'g': 59, 'h': 60, 'i': 61, 'j': 62, 'k': 63, 'l': 64, 'm': 65, 'n': 66, 'o': 67, 'p': 68, 'q': 69, 'r': 70, 's': 71, 't': 72, 'u': 73, 'v': 74, 'x': 75, 'y': 76, 'z': 77, '|': 78, '«': 79, '»': 80, 'À': 81, 'Â': 82, 'Ç': 83, 'È': 84, 'É': 85, 'Ê': 86, 'Î': 87, 'Ô': 88, 'Û': 89, 'Ü': 90, 'à': 91, 'â': 92, 'ç': 93, 'è': 94, 'é': 95, 'ê': 96, 'ë': 97, 'î': 98, 'ï': 99, 'ô': 100, 'ù': 101, 'û': 102, 'ü': 103, '—': 104}
INFO:root:The vocabulary contains 105 elements
=================================================================
Layer (type:depth-idx)                   Param #
=================================================================
Model                                    --
├─Embedding: 1-1                         13,440
├─LSTM: 1-2                              3,416,064
├─Sequential: 1-3                        --
│    └─Linear: 2-1                       65,664
│    └─ReLU: 2-2                         --
│    └─Linear: 2-3                       16,512
│    └─ReLU: 2-4                         --
│    └─Linear: 2-5                       13,545
=================================================================
Total params: 3,525,225
Trainable params: 3,525,225
Non-trainable params: 0
=================================================================
INFO:root:Generated 
>>>
LA GRENOUILLE t9(GTDz8r]ë6okrnjÇQ*ÈNmÇ]NB,xîP!4iÔv6lRÇç'SCabBoyTàq4+é ?F'aâbQêû>/ocG
zC>ov«Ü»;hZWAS:ÊqL CXsô 4-jGu?uIYÊû][ê:ck]+ÊSÔVE'âr0Ràp«:Fvco|T5cDO«|;aÉ(P'vfi+o(AX,»tMDhï<)BçDV9Êd'Çv:ÎI(Y'qTP'PCzrÊmgXfeaXWdÂ?ÔE
<<<
[73,5,95,59,53,76,53,66,72,2,64,57,73,70,2,57,71,68,70,61,72,10,2,1,38,67,65,54,70,57]
u'égayant leur esprit, 
Nombre
[5,95,59,53,76,53,66,72,2,64,57,73,70,2,57,71,68,70,61,72,10,2,1,38,67,65,54,70,57,2]
'égayant leur esprit, 
Nombre 
 [=========================== 313/313 ============================>]  Step:       11ms | Tot:    3s189ms | accuracy: 0.230                                                                                         
Sliding window train metrics: 
  accuracy: 0.22956575351596348
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      142ms                                                                                                           
INFO:root:[0/100] Validation:   Loss : -- | Acc : 34.867%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE naRs le bas le aux. in p're v'ommacat se se ars mot mataica la le sèn in fit vouts dre agcer 
itait que. 
tide sares prAcequi iz gis 
ne que, en jrate qu'ep dit vores-dAUru donstavont. 
 Movriger se lo
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ; en de ge,, mis por fagles dord elce fiqus ce ses, fén ers m'or ca jE-visréant le le gains cit atte é ; en én foucie liec gor sans tietar, mert sern artg'ane drocindi. tenal borvrâ. 
Seces ques son li
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s141ms | accuracy: 0.403                                                                                         
Sliding window train metrics: 
  accuracy: 0.4027977737785777
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      151ms                                                                                                           
INFO:root:[1/100] Validation:   Loss : -- | Acc : 42.764%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE VEXTRALHN APE CIAU- IRSAc TODAT  
A R TAURTAE CE LON EN, PRIQALBARONGANDIMONU « II ET DAUTDE  CEChINE 
D'Rhuerte : étois
Uttir de sain, ins 
Que ne manjaster ec qu'il vous Paétenir entour chade. 
On ma
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE «
LA FAÉEaIfSUOMIX CES SEOfTAb LES DE Od MEMVAhUÉE 
EpOUD LA" A cutoursse santrez; 
L'-lus l'Pencit pour flerverie et ni pait, 
Orquement 
Qui le plloirs 
L'O f«S d'NHIVARD JAIT QUCOG DETTE 
Du d'harai
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s161ms | accuracy: 0.448                                                                                         
Sliding window train metrics: 
  accuracy: 0.448498633606612
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[2/100] Validation:   Loss : -- | Acc : 45.234%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
II D'ANER RES QURCE 
NAATRON EN OBT NANUe LHANBE
PIT-ALMOMCRILRE 
Ce n'et pas en M'ons rylaux dites une mainducte toutes sort d'énanfens donneur averta glangt, des noumant pour vous flant tôt aurrivus
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
LE CANQUinsons un-moin les il érides 
Temble tout ; on ces plus ne toutes mon conteures de plès vompen. 
Lont migneur sacidesses. 
D'ort je rabafhas l'aurant-at à cépôtoit dit mis que n'avoice à vifus
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s160ms | accuracy: 0.468                                                                                         
Sliding window train metrics: 
  accuracy: 0.46752316203425975
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      145ms                                                                                                           
INFO:root:[3/100] Validation:   Loss : -- | Acc : 47.165%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
NTT.
Adieux.
X, Houffrire, une tout, Bout peau voulait sourt. 
Il tout une otci dans quoi son cette bes vase commes auvoir faire, pour l'avec le Chas. Saise, heurer nb ces saveils à maîtresse souvant 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET DAU US 
LE NUinellaaon, apslessiez itelle à les hélorans qui l'actant qui coûsons
Lontés sans des ri'nou, prab étaire. 
Son Vitiez sagesse
Du pai; 
Le dépoquer Derect. Ainles deux possiers, y passe,
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s152ms | accuracy: 0.479                                                                                         
Sliding window train metrics: 
  accuracy: 0.47864260481237075
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      144ms                                                                                                           
INFO:root:[4/100] Validation:   Loss : -- | Acc : 47.646%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET ARD LE SON ET LE CEITOT MITE  ETANLE 
LE LIEIPRE.
J'en frant, on Polins misent d'arbelle, 
Que l'oment qu'il en bonder qui ces mot, mais vis. Avait ses coueur ennut, mes mab. 
N'opmonstes s'en perge
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
IX 
LES LA TONE ET LES ME EXDE ET LE LISAMASBE DES ETETS ET LEE TEn 
Avant qu'il content 
Pour pin le moment bot? que ne monsgez-moi. Il l'Emmosez 
Son baccève je vau lonner qui vous maindre de Chavaî
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s180ms | accuracy: 0.485                                                                                         
Sliding window train metrics: 
  accuracy: 0.48522462174231795
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      144ms                                                                                                           
INFO:root:[5/100] Validation:   Loss : -- | Acc : 48.372%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DE.
Veux et sait point à et ceux pl'Hroce n'éclaîgeant pouvez de peines Ravôt'at adanté qui ne se trigres à charnés de nortimens au Lien fut la mire tro fers y tombeaux maisosie de ration du plus arbre
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE EVOI DE OLMURAG 
LA MELIRE GEVEGGOR ET LA GAL[ 
LA II
LE Eu d'pléfaté bien là l'ée
Fort. 
Tait la jourdée voeut sur les heur ? 
Nous suivo iVÂMou de sujevelhine fa negd qu'entre un garder a ce l'air de
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s176ms | accuracy: 0.491                                                                                         
Sliding window train metrics: 
  accuracy: 0.49122342198227054
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[6/100] Validation:   Loss : -- | Acc : 48.764%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET TINE 
LA MAENE UNANE LE MELIGE 
Un a sont faucaraire de dous, demeurée de à l'aciez-vous sa coursus. 
«
LASÉSE MES MUUIANE 
LE Il ne fereux marté en vous ont par certais crune à sourtirent ne Caisai
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE le cas vous solieunaient on purule; 
R'ordonquait contrement caissant l'enfen les chrêner le pars au vaîtôt dire ont fait la leraient plus une embeur le vous, elle pas songe faut les hautes, et que la 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    2s845ms | accuracy: 0.496                                                                                         
Sliding window train metrics: 
  accuracy: 0.4958724921682331
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      149ms                                                                                                           
INFO:root:[7/100] Validation:   Loss : -- | Acc : 48.237%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DARD PITORI.
S'NÂT, LE.
Il ne né fut la jour: "
Lui fournit grandes êtes avous a commune camis.
Sur coutaime et micres fautes les autorut oeucs du Semmes praix. Je sais et mort bot ne la grande tout le
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE À BÉET LA RICE DE LON
Un nor rit porte ce que vous la teluë en douce on et que que pournit; il chère où rien. 
Ne point la mord aux y comme des abissoutait epfèta s'il comme ces êtiles tout ce malinarc
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       11ms | Tot:    3s159ms | accuracy: 0.499                                                                                         
Sliding window train metrics: 
  accuracy: 0.49928680930480535
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[8/100] Validation:   Loss : -- | Acc : 49.516%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE LEE POM(STIN PAIS ET LE DRONNES 
LE LIFLEF LE LE CI 
[Émalphagez lui 
Nes plus riant ne fit une le Nous, mais que vous est les êtres  ici-êtes la journait-il tous la curie, elle telle noir devueds, à c
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DE SENDESSALIFAN ET LE LI LOUIGRE À DEu Bline. 
Luy, luy, ridysante prend bien tous futes et tout, ! la Pereins, le homme lets prétend car le maître, sur lui que l'instru'extrats, trouva fronger ner se
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s160ms | accuracy: 0.500                                                                                         
Sliding window train metrics: 
  accuracy: 0.5001716323401985
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      143ms                                                                                                           
INFO:root:[9/100] Validation:   Loss : -- | Acc : 49.123%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET DE MANAE ET MIDÈGE BÉVE- 
Moires guerres. 
C'est eut jamais, puy si chassée de hhiacun empos au perte par moi 
De pli. ennemis aurez egglre Hilement dont El était les voilà jamais une ragle lamptte.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE JÉ[HEÉEETWES DE HQ DE ET LA RELUE ENQUÉVONS ENSREDNRE, TÉMJE. Quand trouvez de me femme auvons de s'avant 
Quelque et mon sachudeule vos allait cegent se frapme dont l'élique 
Sage, qui la prais où j'a
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s158ms | accuracy: 0.518                                                                                         
Sliding window train metrics: 
  accuracy: 0.5177847763780578
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      141ms                                                                                                           
INFO:root:[10/100] Validation:   Loss : -- | Acc : 51.370%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Le pris se tut, tourtuce l'ésratement soue l'autel par où qu'il vivroire passe Prouve toix juste, dont répondez à la dicces de que rave qu'il ce tourné les ; 
Quand il trouvetreau, lui qu'on le galor.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE FIELURSRON PHAT D'ET D'ÉTACOLUE DE RENACISUTAINE 
Mhein, venez-moy.
Ils entend l'Éage puissuèle; 
L'entre de grand campagne ont cette à Blage, lui hête le crit qup qui des pays rendlaiis Louette 
Lui l
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s154ms | accuracy: 0.526                                                                                         
Sliding window train metrics: 
  accuracy: 0.5259364793707926
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      141ms                                                                                                           
INFO:root:[11/100] Validation:   Loss : -- | Acc : 51.796%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ENPHEL 
Éjais un patte citié dans les charnés.
D'être au virquer, chemigner paruchait
Le contre queeur la retire nadaures, à quittez ainsi fit qu'il perte marbre que que ce qu'étonge, 
A bien? Ceux bie
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE LE SIN
FINFACS ET LUFETUGE 
[É, Avez plus de l'aurait des falles corps damès voulus d'reçant saison, il eut maint découvrer fils:couronné.
Torte ;
Dit-on de Même cherchait sa voix d'une rosent notre gr
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s158ms | accuracy: 0.530                                                                                         
Sliding window train metrics: 
  accuracy: 0.5299723388655605
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[12/100] Validation:   Loss : -- | Acc : 52.110%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET L'INENUTACOUT ET L DES XOUT D BÈISE 
Ce sauvage au travaire et pas avait 
Constant trop voisin encore le Pégula plom j'avoit sa Tuufonne atteinte 
Rendit ce que je ne faisse au consentueur, s'expris
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
D'air à ce corps l'on tous l'ouvein al! j'agg bongeait l'Éleveux je peu de ton bâillon. 
Vlains-moi qui ne se derrant en réfistant suspendre pas à joute
Ne son état sicn réfend. En dans laisse à plus 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s162ms | accuracy: 0.534                                                                                         
Sliding window train metrics: 
  accuracy: 0.5338082383523294
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      143ms                                                                                                           
INFO:root:[13/100] Validation:   Loss : -- | Acc : 51.712%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LA LATCÉDE SURUT ET ARPON 
La maison jalouse en 
De ces, dis ! ces ressed des traits de vied, 
Repentibs, le plausèrent de prince. 
Tant relle ne suit, nous emplimage est bien réjenait un odéséritie
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE L'ERIGORE 
Les parce.
Vrai te fureur de Cormu éplace et vous sa moutand, dit-on. 
 
OG", 
Mais lui à son nuage si pour une prenait tout en du chose en soit Be corps. 
Dépar quand il a'exemple en vous, 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s152ms | accuracy: 0.536                                                                                         
Sliding window train metrics: 
  accuracy: 0.5361511031127111
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      148ms                                                                                                           
INFO:root:[14/100] Validation:   Loss : -- | Acc : 52.018%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LA MASPRRÉ CHÉARCORÀ.
Vous sens.
Puisque toujours cheval plus; con jémon ne muifque ; le bout-il se voyait remeurs Lunesse: comme indigent ;
Vous devenir dit-il, renquise, et les gemfors avec tant d
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE DOURNRAI 
ET SOUMUIS ET LES DOUGS 
Non mai, une ojt.
Le mentir de s'en bruit? Homme sens en ai prés des douleurs, et le roi mon hérier eut pour le suffâte !
Ceux entendens de rend vain et de ses 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s159ms | accuracy: 0.538                                                                                         
Sliding window train metrics: 
  accuracy: 0.5382773445310935
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[15/100] Validation:   Loss : -- | Acc : 52.490%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE  CORNIGON 
CE. LE RENOURS 
Et je me mes deux esprits vos jongs mit en fois, on ne le Ciel et le poél
Ils se s'étoit sarquet blondix et en son foyent et pas trop rivant; pour campte peuple s'en averti q
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ANT UUSRISSASUON, TIu vivoüe, 
Ces abà de poisson, leur jeter du memplin, qu'à frisant confonde ; ce proche
Si la plus vous a point l'air le plainte dans l'accomptes, les maint chez vos couture avé; 
A
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s140ms | accuracy: 0.541                                                                                         
Sliding window train metrics: 
  accuracy: 0.5408818236352735
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      144ms                                                                                                           
INFO:root:[16/100] Validation:   Loss : -- | Acc : 52.205%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Condit'aimaire l'un; il vous goût de sa trouverait des taercours en les jours, 
Les être ; la roi de tant ra cette grand pacelle qui prérira 
Où cette sedelle de quin, Humerait vous fait dans l'ludent
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE EPHILUTRE JEECREAN 
[Phinage à lit, frise. 
Onfin dont que tout serpent, dans l'OR>f
Une litre charboutier le cirance craignez plus du fait fort d'un Hibouvaux eût un plus grafi plumer nous avoir, et q
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s130ms | accuracy: 0.543                                                                                         
Sliding window train metrics: 
  accuracy: 0.542766446710658
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      144ms                                                                                                           
INFO:root:[17/100] Validation:   Loss : -- | Acc : 52.878%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET VIEUR LE COUR 
[Proit l'honneur. Elle n'élrupage. 
J'ai ne suivois si prit qu'au regardine au, innocente, 
En fut se plein sa tire, soi de voir ici s'en coup ne nuite, 
Ne fureurs qui se tout à la t
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE  XICE ET L'ÂNE ET LE lui à intrépide, 
Ce respect de voulut achrangez gueule devient faire à ce la scère ; les Rois où il saurait assautre étion par l'agit à la dose étouche lui qui demeure voulez 
Tou
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s163ms | accuracy: 0.545                                                                                         
Sliding window train metrics: 
  accuracy: 0.544502766113444
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[18/100] Validation:   Loss : -- | Acc : 52.534%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Là des hommes ouverts, dit-elle. Il tendait andoune à qui conseil ouvreur pieds l'insolelle de l'autre la porte et d'un Rats 
Qu'il ne s'en n'est de l'élland qu'on n'avoue 
Àson, se voyez ouï le vint-
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Sans soin fait la fuiesser déchilessencier.
Des consteurs l''jour, raison que je conseil; mais vous verme ainsi qu'on ne nous assez époux? Je rendon, s'il avente,
Sous si tout c'est ouvrages 
Qu'a pas
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 153/313 .............................]  Step:       10ms | Tot:    1s536ms | accuracy: 0.547                                                                                         
 [=========================== 237/313 =============>...............]  Step:       10ms | Tot:    2s395ms | accuracy: 0.546                                                                                         
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s172ms | accuracy: 0.546                                                                                         
Sliding window train metrics: 
  accuracy: 0.5455625541558353
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[19/100] Validation:   Loss : -- | Acc : 52.829%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Gisons: il ne fait coupez s'il vous éblage à charlons a plutrement, 
Ayant qui faisait un goûte ses dieux égards de srate homme partie rend
Sans dévrail en un Sacherais les pleins, défaméa ordut et ge
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE RURATANNEUX ET LE LOUNE DEURAN.
Éprouva celle qu'il n'est folen blond pro de maudite versemp insaire la pleine, 
Disons sa plaît par le platon argent difficun esclat : 
Les fiances, et liqueur co
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s170ms | accuracy: 0.558                                                                                         
Sliding window train metrics: 
  accuracy: 0.5579717389855363
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      145ms                                                                                                           
INFO:root:[20/100] Validation:   Loss : -- | Acc : 53.959%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LA DACOÜDONS 
Ce que n'est doux Zédèlurent y contine à tourment, ressurent l'autre tirée : jeunir, se faite tous l'amour s'étant tant de pourpre, si sa favorable, 
On vit. Le pronapie à son besoître
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DE LA SÉVRE 
Chez vous jetait odieux; 
Ils feraient plaître toute louche et point qui ne te laisseroit l'années de huit sa passe à son s. Lui dédestueuse déplorer aussiare Boussent l'Arryle ; ils me vo
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s162ms | accuracy: 0.565                                                                                         
Sliding window train metrics: 
  accuracy: 0.5651086449376792
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      145ms                                                                                                           
INFO:root:[21/100] Validation:   Loss : -- | Acc : 53.989%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ENLAT DES JEUNE MARD ET LE MARDONS DANLE ES ET CECE DE LION SOURAN ET LE GENGES ET LE CHÉDES DES IV 
L'ET XVIIEÉ.
Il aurait maint.
Honneux 
Il m'en en devoir enfin. Vôtre fait mon Dieu sur la sont chan
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE SI 
Sembla sur le Cieux et choisit pour corve j'ont l'amour est dont d'acconsourde. 
Je ne semble et chaque heureux était faire émagisserais les Rois inruit que l'Enfari, toupureuse pêcher par leur end
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s156ms | accuracy: 0.570                                                                                         
Sliding window train metrics: 
  accuracy: 0.5703359328134366
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[22/100] Validation:   Loss : -- | Acc : 53.941%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE RENE.
Son incon 
Tous laissons qu'ayant périt; et pour mon que la célonce, qu'il vivier de ce séjour, s'officiez pour moi. 
L'on est de voir ses progritions de son 
En acquis on a fermeur fond de
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ALTUNLAMAILALI.
Si vous sans le laisse, quelquefois qui vous ne nuit, ne les cris pris à chaque cordin sur l'entend de moi. 
Mis toujours assort le plus si bien pense, 
Compagnie ? Ma. Mimpleurs? Nivre
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s140ms | accuracy: 0.574                                                                                         
Sliding window train metrics: 
  accuracy: 0.5737719122842102
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[23/100] Validation:   Loss : -- | Acc : 54.104%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Sans la préus n'avait reste à l'autre ait du même que la favetant de Magée qu'il le sel avait avartive à me montôt Compagnon ;
Qu'il montra devenu aux pays 
A ces larmes ressorts du grand instreux eut
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DE REM-CIDISXUX 
Si les Corfevers je marchais un historruait; babillon que je ne ne porte.
Qu'en du Filence à sa fribité 
Retors. 
Quelque bout n'elle vient donc j'avançais en lois, impontenir réduite,
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s153ms | accuracy: 0.578                                                                                         
Sliding window train metrics: 
  accuracy: 0.5777227887755778
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      145ms                                                                                                           
INFO:root:[24/100] Validation:   Loss : -- | Acc : 54.214%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DE L'HOMLOGNE ALVILLE 
PON PIEIGTESSET LE LOUP VIRD ET LE BOYÉSOIS DE MAPON UN CHAIBLE ET LE RAT.
De père avec respect de croyât de bord. Non rétends sur les sang avec un grande ce diste traite! 
A la 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Ce qu'a bien concorte soin dans de nos rustes qu'il confesse aussi, à l'Aches à témoignance. 
Prenait à Florise à liser, il vit selon les croqueurs de l'abord, soutienr. 
Des Femmes, on s'en étonnaien
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s157ms | accuracy: 0.581                                                                                         
Sliding window train metrics: 
  accuracy: 0.5811587682463514
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[25/100] Validation:   Loss : -- | Acc : 54.264%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE À RAR, UN » 
Ce n'était un ritue jaloux de ses défont un cheval injustée avec non Alors, cependant vous faire objet dénente,
Faître doux autr ;
Vos mieux rendues, et le vit plus foi qui fut l'autre bât
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DINMEVEINE.
Jeille la qu'elle le décaime. 
Prenons-nous, pour son île eût la tas encore jour. 
Je lui dit cevoit l'Elâtre; et rien qu'roiard apris: J'ai parmi se supolement, aux hommis. 
Ce menait le F
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s169ms | accuracy: 0.584                                                                                         
Sliding window train metrics: 
  accuracy: 0.5843714590415253
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      148ms                                                                                                           
INFO:root:[26/100] Validation:   Loss : -- | Acc : 54.343%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Entre ces deux bras votre nêjet du Bramin de plus touche des innocents. 
Un parsée de la nature sur la Tortue ahquitté qu'en long aux temples que son tremblet. 
Supporter les beaux esles amanne ma méd
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE COVORE 
Deux Ainsi beaux mal de Diffaire faisait blesse.
STANCES 
[Caron sérieux fait. Lui tout est des sins se serpent. 
Lui bien mal alarbice avec presse un mysté de Belant il est de ces gros assassa
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s181ms | accuracy: 0.588                                                                                         
Sliding window train metrics: 
  accuracy: 0.5876058121708992
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      152ms                                                                                                           
INFO:root:[27/100] Validation:   Loss : -- | Acc : 54.334%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE Sude qu'il vivait naturel de ton. 
Ce deux ce Liéx d'une hasas.
Il n'est, 
Par un fort d'un grande solité douce, il ne sont ne disaient me tenir qui composement ces tambours faisaient l'aide plus cette
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE SINGE 
Un peu s'ayez rien du délein. 
Un chose, à la fonté 
Sans l'emporte. 
Ce long plurie tombez le sorte d'abord comme un tant des livres mis de la procis d'un objet dans son delà l'autant que donné
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s156ms | accuracy: 0.590                                                                                         
Sliding window train metrics: 
  accuracy: 0.5900803172698795
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      148ms                                                                                                           
INFO:root:[28/100] Validation:   Loss : -- | Acc : 54.067%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ENCESSE PETINNE, DE LASACRE DE QUITE, ALSAND ET LE ANETTOGN'AVEAUT 
Vouloir bord 
Des Lymphes prenait cruel ajoutient autre grâce en toujours désert, un esprit, si tout vôtre passant dont vous coix les
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE À RIRILGONSRÉ PAUX ET LES DLÉSUUE ET LE COMFEUX 
PARYIT 
Un point un fils, et vers vous, à son sein par ceux qui ont en divinité, ou témentait. 
Pour blanchissié de le mauvaisant la Malelol : 
Tes sanc
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s172ms | accuracy: 0.593                                                                                         
Sliding window train metrics: 
  accuracy: 0.592921415716857
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[29/100] Validation:   Loss : -- | Acc : 54.316%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DANRE 
Un Sitte à brouer un autre mis en méché. Des-pétieux et les rit dans le vérisser 
Un amant 
Des expart notre Rattrapera. Pour rendait quehelés étaient une courer. 
XI 
LE COINS 
Un riche à tire.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE " LE PIGOGLE 
Le frais vous, il l'âge vaît contraissent de tout le coup dit que le brave.
Quelque jour par devant ce Baisez conduisé 
Pour bouche redoutez-lui d'entendre des flatteurs d'entend ces moin
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s191ms | accuracy: 0.607                                                                                         
Sliding window train metrics: 
  accuracy: 0.6070769179497435
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      144ms                                                                                                           
INFO:root:[30/100] Validation:   Loss : -- | Acc : 54.528%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE À BÈNTLÈRE 
Un Homme s'il vassion, il les Sourides ingrats.
Si l'avait vître faire tous les morts, les divers sont autres véritiés ;
Quand la guerre enfin ; la Reine habille un baché sagesse jalou? Voi
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DANGESSE ET LE LION MÂNE ET LA MOME 
Paillal fondonne tous les plûtes défauts et qui fut dit 
Du france : rien que le prêtcère entre le Thisbé du rapinable ses sujets 
Que l'Anze! ayant corps les forêt
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s191ms | accuracy: 0.615                                                                                         
Sliding window train metrics: 
  accuracy: 0.6151586349396784
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      150ms                                                                                                           
INFO:root:[31/100] Validation:   Loss : -- | Acc : 54.507%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ALISUR. ET LE BATE DE SON SES FALBELLE 
A reine déterle à l'elle qu'il devait le coeur on vos corps tombent, et tombin: 
En faisant-ils droits la longues et pour elle a pris l'actace, c'était des compt
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE LA PAR DES TRAT ET LE RENARD 
[Ésope] 
Ils raudoits discrètes d'un vigoureux, aventure de biens de courtisan. L'enrhier 
Victe de ces pères une tels cours en l'autre, et qui suffit. 
Tout autant du Fot
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       11ms | Tot:    3s187ms | accuracy: 0.621                                                                                         
Sliding window train metrics: 
  accuracy: 0.6208774911684326
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      148ms                                                                                                           
INFO:root:[32/100] Validation:   Loss : -- | Acc : 54.298%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE MACRESSESSINTAINE 
Teurez que l'abat en bien fait sous un cheval que sa court, et puis cette roi vos cantons !
S'offrant eu chacun peu d'aucun du plus ou peu dans les Renards à son émanches, compère a 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DLUCRE ALAS.
Olente à sa La compte une tête écarté qu'un tigement 
Peux-tu soucis de meilleur et les Savaient l'épouse qu'avait une petite l'effet d'un y baisé.
Animal le boyaume un regarder travers vo
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s176ms | accuracy: 0.625                                                                                         
Sliding window train metrics: 
  accuracy: 0.6254949010197958
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[33/100] Validation:   Loss : -- | Acc : 54.260%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Trop plus dans les bourchilots il fantaisie,
Et sans concevoir perdit l'autre, dit-il voulut fait dit: Cypendant la Compté de tout, persant le bravera ces fers l'assis le croire. 
Aussi précieuse il n
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Un si. Un Moutrage ; tes personnes notr'Epichus soit saurais mal homme d'accourons fleuris, 
Fit riche est grossier huche à charmée du Sématmin, tel disant
Les occasions pour ressent par a bientôt ils
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s176ms | accuracy: 0.630                                                                                         
Sliding window train metrics: 
  accuracy: 0.6303939212157569
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[34/100] Validation:   Loss : -- | Acc : 54.002%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET CERLAET ET LE STEIL
LE RATS 
ET LE CHAR. 
Un troute fait avec mon vers lieu déjà sur le nuisant plus d'un beau pait découvert
Qu'être un bons auprès du monde et l'autre s'aprience : est-il trouva qu
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET L'ESTES 
Dans la maison de qu'ils rit, il se regarde pour a mal? 
Si la porte lui dit-il, il perce des Compagnons aideroupeaux que je sais quand le travers voix avait auprès du sotte molonge, auraie
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s164ms | accuracy: 0.635                                                                                         
Sliding window train metrics: 
  accuracy: 0.6347763780577215
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      142ms                                                                                                           
INFO:root:[35/100] Validation:   Loss : -- | Acc : 54.039%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Était voulu coupe au Portier, agréer partout l'autre se guignée enmain prunage, et l'a délivré de lui supprime. 
Voici l'histoire dans l'assembler à tout ce n'aurait est bonne: 
De songez allait un gu
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE MAMBOYUNÉ
Cependant vous a peut s'approche ingénique : 
Il entière et l'étrangler clorant portement 
Grésus de l'honneur de tout ce mal aventure 
Du moins prenez, quand lui acheçons d'un meuble crédit 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s166ms | accuracy: 0.639                                                                                         
Sliding window train metrics: 
  accuracy: 0.6388688928880893
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[36/100] Validation:   Loss : -- | Acc : 54.104%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Un Hirondelle avait aux simples contre le jeune matière, 
Troupe: on se logez-moi qui est fiait jusqu'eux, par le pauvre Chevaux conseil, on faut bien venir au désert,
Aussi courtisan beaucoup de mour
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE RENARD 
Dy, Elle s'en veux transport.
On aisément son dernier 
Que l'Histoire à ses fameux mangeaient que vous m'en contrement 
Ce que chériver mon frère; ils étaient perces, dit mon, tant du Gad
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s161ms | accuracy: 0.643                                                                                         
Sliding window train metrics: 
  accuracy: 0.6433563287342533
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      144ms                                                                                                           
INFO:root:[37/100] Validation:   Loss : -- | Acc : 53.826%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DUSTANT ET LE BAPIMET PROUR ET L'ÂNE MOMME PUNALESUDE.
De biens point d'exemple en égal, c'est l'objet, comme moi, que l'evenir. 
Aux personnes de quoi décharger tous les chats cent ôter un censeur, qu
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE SOSI, LA PMENE.
Qui veut plutôt que chez les Courts, et charme le plus prompt fut personnage sur toute talents 
C'était diable vivant.
Lui qu'elle aura battre grère
Qu'elle retrouve à notre sotte fait 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s158ms | accuracy: 0.647                                                                                         
Sliding window train metrics: 
  accuracy: 0.6474388455642203
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[38/100] Validation:   Loss : -- | Acc : 53.635%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LES BERNHANT DES MASÉTE EIX CEILANT DE LA MIELES 
[Ésope: il y supplie; 
Quand Iris, tous l'eusses ainsi le trouva donc creçlant finir longtemps leur père; 
Tu te préfère allait beaucoup de son libe
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DES DEUX SOURURES 
Disons forts, car il dirait sûr de langage. 
Toute mort ne sont couronnants, elle avait une vizé fait de Soleil pris, une ; le Prince en plusieurs malheurs? 
Un homme tout principal 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s154ms | accuracy: 0.651                                                                                         
Sliding window train metrics: 
  accuracy: 0.6513847230553892
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[39/100] Validation:   Loss : -- | Acc : 53.621%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DU CACOIRE 
D'ÉPINTANT MERIl cris à flonça dans tout le voisinage, 
La Fontait vile alléguait les yeux de ton. 
On fatée une Pèlerine. 
Pour vous êtes jamais étaient tout marché vous en suspens. 
 
LE 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET L'ALOURES 
É 
L'HENOURIS AUS DES ET LE SOLITE ET LE BALANNES 
Un amoureux vous trompée eût pris que je le fais gloire femme à changeait de quelques-uns divers conshons du puser ont un mal de l'oeil 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s168ms | accuracy: 0.664                                                                                         
Sliding window train metrics: 
  accuracy: 0.6636356062120916
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[40/100] Validation:   Loss : -- | Acc : 53.681%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LA MELALTES 
Diffmère, ou de pèlerins, des vaisseaux, et d'un muse à dire, et frend le desseins où les Morts on ne se préfère, quelques fendre un peu souper le jeune homme toutes les puissans 
Au pr
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
SUSELÉTE ET SES PÉSES 
[Ésope] 
Troupe avec la plure. 
L'enouille du Chat-huant connosta: est soi que je veut au monde, on lui passe aux milieu de soupirs, tenait à quelque présent; un brasier ennemi.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s157ms | accuracy: 0.670                                                                                         
Sliding window train metrics: 
  accuracy: 0.6698327001266419
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[41/100] Validation:   Loss : -- | Acc : 53.553%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE HILON ET LE VIEILLARD DANS LAIT ET PRON
L'avare humaine y meurt, au nez entendu l'un en voyant la revons pronondonnance fa fait amant:" Il mais qu'il sillon 
Voudrait raffoire. Il était rien d'en
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE QUI LA BÉVRE ARUS 
Un dommage, on voit en prison.
En s'adresse à peine couvert de père cause d'Auguste atteint. 
Jurais-je Netor fronnêts. Écoutez-nous deible se consoler la noiser : sa pays, les ouvra
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s166ms | accuracy: 0.673                                                                                         
Sliding window train metrics: 
  accuracy: 0.6734019862694128
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      147ms                                                                                                           
INFO:root:[42/100] Validation:   Loss : -- | Acc : 53.417%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE CHASSEUR LE MAYER 
[Ésope] 
Et comme son Essomme une avis dure 
L'autre piqué : il préside, voyant d'un Delé dont le charge aux grêceuses en homme qui était pas les flancs qui désignaient dans l'
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LES POIS 
Plutôt que moi de son guérisse à Chasseur, chacun est versèrent de ma fourmi de partie 
Plus de charnier et furent plus de terre, aimable. 
Quelque hameau. 
Donnez votre léchira d'argant p
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:        4ms | Tot:    2s419ms | accuracy: 0.677                                                                                         
Sliding window train metrics: 
  accuracy: 0.6767213224021866
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      133ms                                                                                                           
INFO:root:[43/100] Validation:   Loss : -- | Acc : 53.365%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE DES LILUMALILAGOGOGE 
QUI SON THATANGE 
Tout son ronde à moi, dit-il, dus fausses anciens habitants; pour Benté 	formité au loin, 
De ces pays-làment ses meilleurs inventions enfin lippher de vous un p
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
JELLE 
Pline les Perses de mit que chacun n'est fait la déplore, le roi seroit pourquoi qu'il vous passion.
Elle avait une charait, comme il donna possiblement débauche de moy l'attrabler les charmant
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    2s234ms | accuracy: 0.680                                                                                         
Sliding window train metrics: 
  accuracy: 0.6799140171965609
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[44/100] Validation:   Loss : -- | Acc : 53.325%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE BOURLÉE SE PUCRE ET LE COQ 
Un héros de reçut à la faisait son leçon des efforts de Pâris, 
Des chefs à ces clartés d'un Cerf tombé à ce poil, 
Honorent de noyables quelques-uns ennemis qu'il se ronde.
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET LE CHARTANTE DE DOSONGE ET LA PÉQUESTE.
Quand Jeune ai pris
Dit entemps à ses personnes nous plaindre dit-il, saluté se croit accru Saint homme de loin son effort donc sur un corps demandait à ce qu
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s162ms | accuracy: 0.683                                                                                         
Sliding window train metrics: 
  accuracy: 0.6828800906485369
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      145ms                                                                                                           
INFO:root:[45/100] Validation:   Loss : -- | Acc : 53.213%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
Cependant le Fils dites-moi, sur le monde avec cette vertu ; mais elle trouvé si placer l'âme et la douleur est envie,
Et quand le saïez frateur :
Mettins, il la lui croit un peuple saisie, 
Puis en l
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE ET CERLE D'UNE ET LES BOURTE RENARDS
Droit, suit en abus; car une Palais, et n'est bonne-ggence à vous comble de coûtier cette plainte.
S'ont la tête de telles divinités les âmes, ces traits; un grand 
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s142ms | accuracy: 0.686                                                                                         
Sliding window train metrics: 
  accuracy: 0.6860527894421117
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[46/100] Validation:   Loss : -- | Acc : 53.117%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
D'ÉPONTANT XILLE ET LE PETAMINGE 
Ces peuples tours quelque faut jamais aucun que femme enfin immainte nous peut suffix, 
Autant qu'il aura pourpre : 
Je là ; un ridicule comme il passa des emplois en
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE FUMAIER 
La puisse tôt prendrais craint. Al
Dangeurs pour nous mot, lui était abeilles chez eux limes à festin. 
Il suh donc d'autres Mercures maltl'oubliait les plus belles, elle aimoit: « Mais, ayant
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       10ms | Tot:    3s148ms | accuracy: 0.689                                                                                         
Sliding window train metrics: 
  accuracy: 0.6886722655468901
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      145ms                                                                                                           
INFO:root:[47/100] Validation:   Loss : -- | Acc : 53.136%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE 
[Ésope] 
Un palais de fleurs déplaisirs les amants les plus dire, et la cause-t pris le même Saint cers de la nhasseur et n'osait l'amant. 
Le sang d'un demi-pour locement de l'èchesse et tout fière c
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE À LA CHARÀ 
[Ésope] 
Elle en mon inventions pleines que je suis être des hommes : 
Ni crains tous les Faunes que vous déplorer votre diable, 
Je font sans exprimait, osé son malheur
Qui sont qui étaien
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
 [=========================== 313/313 ============================>]  Step:       11ms | Tot:    3s157ms | accuracy: 0.692                                                                                         
Sliding window train metrics: 
  accuracy: 0.6915066986602676
 [=========================== 79/79 ==============================>]  Step:        1ms | Tot:      146ms                                                                                                           
INFO:root:[48/100] Validation:   Loss : -- | Acc : 53.113%
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE PAON 
Selon de vous-hetin
Avec toi, Jupiter, dit-il se prix. 
Dieu galant leur brin donc parler sur le faîte, vous manqué 
Découvrit le Chien; c'est la jeune ami, qui paraissaient : J'abandonnais que d
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
INFO:root:Generated 
>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>
LA GRENOUILLE le fit sûr de quoi de non, 
Grûbit et furieux, par quelque y porte, 
Et faisant l'en personne obliger ce put père, avec tant d'arrêter la main, je voulais qu'il n'y contrée vint lui que Escante l'autre
<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<
```
