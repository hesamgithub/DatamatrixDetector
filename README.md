# DatamatrixDetector
Snippets Describing a Simple Data matrix Detector using a heuristic approach

This is a very simple almost toy project that I had done a while ago for detection and decoding of
datamatrices.

here is a brief description of what it does in plain english:

It looks for squares (all datamatrices are square shaped)</br>
Cuts and warps them.</br>
Assumes all the squares are all datamatrices !</br>
Cuts them finely before reconstructing them.</br>
Based on the above mentioned presumption tries to reconstruct the squares as datamatrices.</br>
Once reconstructed, passes them to pylibdmtx decode method to see if it contains someting,</br>
and that is pretty much it. </br></br>

I have included the class that does the heavy lifting (all that is mentioned above) however, you might want to rewrite that based
on your use case, camera API, etc. I had tested it ulsing Allied Vision cameras and it worked pretty decently.

My test settings were:</br>
Allied vision stingray camera,</br>
Infrared lighting,</br>
A Core i5 Windows Machine,</br>

My test results were:</br>
About 20 frames (1,280 x 720 pixels) per sec using camera Async API</br>

It was a learning project for me. I'm putting it out there so maybe another learner would learn from it.</br>
that's all folks!



