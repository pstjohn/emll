import cobra
import numpy as np

model = cobra.Model('mendes')

M1 = cobra.Metabolite('M1')
M2 = cobra.Metabolite('M2')
M3 = cobra.Metabolite('M3')
M4 = cobra.Metabolite('M4')
M5 = cobra.Metabolite('M5')
M6 = cobra.Metabolite('M6')

A = cobra.Metabolite('A')
AH = cobra.Metabolite('AH')

model.add_metabolites([M1, M2, M3, M4, M5, M6, A, AH])

R1 = cobra.Reaction('R1')
model.add_reaction(R1)
R1.build_reaction_from_string(' --> M1')

R2 = cobra.Reaction('R2')
model.add_reaction(R2)
R2.build_reaction_from_string('A + M1 --> AH + M2')

R3 = cobra.Reaction('R3')
model.add_reaction(R3)
R3.build_reaction_from_string('M2 --> M3')

R4 = cobra.Reaction('R4')
model.add_reaction(R4)
R4.build_reaction_from_string('AH + M3 --> A + M4')

R5 = cobra.Reaction('R5')
model.add_reaction(R5)
R5.build_reaction_from_string('M4 -->')

R6 = cobra.Reaction('R6')
model.add_reaction(R6)
R6.build_reaction_from_string('M2 --> M5')

R7 = cobra.Reaction('R7')
model.add_reaction(R7)
R7.build_reaction_from_string('AH + M5 --> M6 + A')

R8 = cobra.Reaction('R8')
model.add_reaction(R8)
R8.build_reaction_from_string('M6 -->')

S = np.asarray(model.to_array_based_model().S.todense())


def reversible_hill(substrate, product, Modifier, Keq, Vf, Shalve, Phalve, h, Mhalve, alpha):
    return Vf*substrate/Shalve*(1-product/(substrate*Keq))*(substrate/Shalve+product/Phalve)**(h-1)/((1+(Modifier/Mhalve)**h)/(1+alpha*(Modifier/Mhalve)**h)+(substrate/Shalve+product/Phalve)**h)

def ordered_bi_bi(substratea, substrateb, productp, productq, Keq, Vf, Vr, Kma, Kmb, Kmp, Kmq, Kia, Kib, Kip):
    return Vf*(substratea*substrateb-productp*productq/Keq)/(substratea*substrateb*(1+productp/Kip)+Kma*substrateb+Kmb*(substratea+Kia)+Vf/(Vr*Keq)*(Kmq*productp*(1+substratea/Kia)+productq*(Kmp*(1+Kma*substrateb/(Kia*Kmb))+productp*(1+substrateb/Kib))))

def uni_uni(substrate, product, Kms, Kmp, Vf, Keq):
    return Vf*(substrate-product/Keq)/(substrate+Kms*(1+product/Kmp))



A = np.array([
[-31.4 , 4.41   , 0.135   , 0.313   , 0.31   , 0.135    , -0.424  , 0.97],
[-2.47 , 0.0608 , 0       , 0       , 0      , 0        , 0       , 0],
[-17.4 , 0.219  , 0.0831  , 0       , 0      , 0.0932   , 0       , 0],
[0     , 0      , 0.0147  , 0.0268  , 0      , 0        , 0       , 0],
[0     , 0      , 0       , 0.00278 , 0.848  , 0        , 0       , 0],
[0     , 0      , 0       , 0       , 0      , 0        , -0.486  , 2.16],
[0     , 0.351  , 0       , -0.0015 , 0      , 0        , -0.0389 , 0],
[0     , 0      , 0       , 0       , 0      , -0.00364 , 0.09    , 0],
[0     , 1.04   , -0.0288 , 0.0859  , 0      , -0.0167  , 0.0993  , 0],
[3.88  , 0      , 0       , 0       , 0      , 0        , 0       , 0],
[0     , 0      , 0       , 0       , -0.713 , 0        , 0       , 0],
[0     , 0      , 0       , 0       , 0      , 0        , 0       , -1.56]
])

x_ss = np.array([1., .996, .164, .0428, .101, .0726, .202, .0202, .0789, .1, .2])
