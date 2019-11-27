# Réseaux de neurones pour du transfert de style d'image à image

Jules Civel, Louis Hache, Younès Rabii, Nathan Trouvain

Projet réalisé dans le cadre du parcours IA - ENSC 3A.

## Principe

### Introduction

Nous nous proposons de produire un réseau de neurones permettant d'effectuer du tranfert de style entre deux images comme décrit par [*Gatys et al.*](https://arxiv.org/abs/1508.06576)

Ce réseau permettra d'extraire d'une image de style des caractéristiques visuelles locales (le *style*) et de les reproduire dans une autre image, qui conserve elle ses caractéristiques visuelles globales, que l'ont défini comme son *contenu*. Ce contenu se trouve donc transformé pour correspondre à un style particulier.

Nous nous proposons également d'explorer le potentiel de cette technique en utilisant différentes combinaisons de caractéristiques, extraites de différents types de réseaux convolutifs pré-entraînés.

### Formulation

Soit deux images $C$ et $S$, respectivement une image dite de *contenu* et une image dite de *style*.
On cherche à produire une image $X$ telle que:

$$
L_c(C, X) \approx 0
\\
L_s(S, X) \approx 0
$$

avec $L_c$ et $L_s$ deux fonctions évaluant la perte entre respectivement $C$ et $X$ et $S$ et $X$.

Un estimateur $\tilde{X}$ de $X$ sera obtenu par:

$$
\tilde{X} = \underset{X}{\mathrm{argmin}} ~~ \alpha L_c(C, X) + \beta L_s(S, X)
$$
