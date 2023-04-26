# DatamatrixDetector
Snippets Describing a Simple Data matrix Detector using a heuristic approach

This is a very simple almost toy project that I had done a while ago for detection and decoding of
datamatrices.

here is a brief description of what it does in plain english:

It looks for squares (all datamatrices are square shaped)
Cuts and warps them.
Assumes all the squares are all datamatrices !
Cuts them finely before reconstructing them.
Based on the above mentioned presumption tries to reconstruct the squares as datamatrices.
Once reconstructed, passes them to pylibdmtx decode method to see if it contains someting

and that is pretty much it. 
I have included the class that does the heavy lifting as mentioned above however, you might want to rewrite that based
on your use case, camera API, etc. I had tested it using allied vision cameras and it worked pretty decently.

my test settings were:

allied vision stingray camera
infrared lighting
a core i5 windows machine

my test results were:

about 20 frames (1,280 x 720 pixels) per sec using camera Async API!

It was a learning project for me. I'm putting it out so maybe another learner would learn from it.
that's all folks!



