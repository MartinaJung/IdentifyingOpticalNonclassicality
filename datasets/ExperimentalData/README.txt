4-pixel SNSPD: 
This is an SNSPD array with four pixels, where all pixels are wired in series. The output voltage amplitude is then proportional to the number of pixels firing (registering a click). 

4-bin spatial: 
This is a multiplexed setup with four SNSPDs.

8-bin TMD: 
This is a multiplexed setup with two SNSPDs and four temporal bins per SNSPD. 

PNR SNSPD:
This is an SNSPD where I use the arrival-time information relative to a trigger signal to gain photon-number resolution from the SNSPD. Given a certain criterium, I allow the SNSPD to resolve 1,2,3 and 4+ photons. Therefore, the possible events are 0,1,2,3,4 (and 4 is then 4 or more photons).

nbar array:
This array contains the coherent state mean photon numbers of the experiments. The length is denoted by D.

P matrix:
This is the outcome matrix of the experiments. The matrix has the dimensions DxN, where N is the number of outcomes (number of bins + 1, i.e., the "0" outcome).

POVM matrix:
This is the matrix containing the POVMs. The matrix has the dimensions MxN, where M is the Hilbert space dimension, truncated above the largest mean photon number that was measured in the specific experiment.

"Which bin information":
This is the raw'ish data, that shows which bins have clicked per repetition/shot of the experiment. The _XX in the name indicates the index of the nbar array, i.e., the response of the bins per repetition for the specified mean photon number. 

"Which event information":
This is the raw'ish data, that shows which event was registered per repetition/shot of the experiment. The _XX in the name indicates the index of the nbar array, i.e., the response of the SNSPD per repetition for the specified mean photon number. 

Load in sparse matrices:
from scipy import sparse
data = sparse.csr_matrix.todense(sparse.load_npz('filename'))

