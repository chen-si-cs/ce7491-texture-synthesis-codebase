"""Consequential algorithm checks; run before interpreting visual results."""
import numpy as np
from efros_leung import Matcher, synthesize, neighborhood_counts

source=np.arange(49,dtype=float).reshape(7,7)/48
matcher=Matcher(source,3)
template=np.array([[0.2,0.4,0.1],[0.8,0.6,0.3],[0.1,0.9,0.2]])
mask=np.array([[1,0,1],[1,0,0],[0,1,0]],dtype=bool)
d=matcher.distances(template,mask)
manual=[]
weights=matcher.weights.reshape(3,3)*mask
for y in range(5):
    for x in range(5):manual.append(np.sum(weights*(template-source[y:y+3,x:x+3])**2)/weights.sum())
assert np.allclose(d,manual,rtol=1e-13,atol=1e-13)
modified=template.copy();modified[~mask]=1e8
assert np.allclose(d,matcher.distances(modified,mask))
center_mask=mask.copy();center_mask[1,1]=True
assert np.allclose(d,matcher.distances(modified,center_mask))
ids,dist=matcher.candidates(template,mask)
assert np.all(dist[ids] <= dist.min()*1.1)
small_mask=np.array([[0,1,0],[1,1,0],[0,0,0]],dtype=bool)
counts=neighborhood_counts(small_mask,3)
assert counts[1,1]==3 and counts[0,0]==3
r1=synthesize(source,(10,10),3,42);r2=synthesize(source,(10,10),3,42)
assert np.array_equal(r1[0],r2[0]) and np.array_equal(r1[1],r2[1])
known=np.ones((7,7),dtype=bool);known[2:5,2:5]=False
filled=synthesize(source,(7,7),3,42,initial=source,known=known)
assert np.array_equal(filled[0][known],source[known])
assert np.isfinite(filled[0]).all() and len(filled[1])==9
print('PASS: masked normalized SSD, unknown/center exclusion, relative candidates, counts, deterministic RNG, known-pixel preservation, completion.')
