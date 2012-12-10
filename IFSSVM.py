from mvpa2.suite import *
from mvpa2.clfs import *
from mvpa2.testing.datasets import datasets

#fake data

def get_data():
        data = np.random.standard_normal(( 100, 2, 2, 2 ))
        labels = np.concatenate( ( np.repeat( 0, 50 ),
                                  np.repeat( 1, 50 ) ) )
        chunks = np.repeat( range(5), 10 )
        chunks = np.concatenate( (chunks, chunks) )
        return Dataset.from_wizard(samples=data, targets=labels, chunks=chunks)


svm = LinearCSVMC()

errorfx = mean_mismatch_error
fmeasure = CrossValidation(svm, NFoldPartitioner(), postproc=mean_sample())
pmeasure = ProxyMeasure(svm, postproc=BinaryFxNode(errorfx, 'targets'))

ifs = IFS(fmeasure,
          pmeasure,
          Splitter('purpose', attr_values=['train', 'test']),
          fselector=\
  	        # go for lower tail selection as data_measure will return
                # errors -> low is good	
          FixedNElementTailSelector(1, tail='lower', mode='select'),
                 )

ds_train = get_data()
ds_train.sa['purpose'] = np.repeat('train',len(ds_train))
ds_test  = get_data()
ds_test.sa['purpose'] = np.repeat('test',len(ds_test))

ds = vstack((ds_train,ds_test))
orig_nfeatures = ds.nfeatures

ifs.train(ds)
sel_ds = ifs(ds)

print "original feature number:"
print orig_nfeatures
print "now feature number:"
print sel_ds.nfeatures


