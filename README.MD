# RGB-MAP: a pseudo-dithering algorithm

This program generates a map between a r8g8b8 color and (as of now) two r3g3b2 colors for use in dithering.
A lot of this code could be better optimized, and the last step could probably be done much faster via compute shading

## Todo

* convert color map generation to compute shader.
* change number of r3g3b2 colors from 2 to 4 (about 5568.5 times more expensive)
* separate base code into library crate
* other color types
* generally clean up code
