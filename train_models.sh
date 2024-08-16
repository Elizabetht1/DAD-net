#/usr/bin/env bin

# Train deep learning model
python runner.py --config-name train_dl model=cnn dataset=BasicMotions hyp.batch_size=8
python runner.py --config-name train_dl model=cnn dataset=ChestMntdAcl hyp.batch_size=32
python runner.py --config-name train_dl model=cnn dataset=Epilepsy hyp.batch_size=16
python runner.py --config-name train_dl model=cnn dataset=WalkingSittingStanding hyp.batch_size=32

python runner.py --config-name train_dl model=cnn_lw dataset=BasicMotions hyp.batch_size=8
python runner.py --config-name train_dl model=cnn_lw dataset=ChestMntdAcl hyp.batch_size=32
python runner.py --config-name train_dl model=cnn_lw dataset=Epilepsy hyp.batch_size=16
python runner.py --config-name train_dl model=cnn_lw dataset=WalkingSittingStanding hyp.batch_size=32

python runner.py --config-name train_dl model=cnn_dn dataset=BasicMotions hyp.batch_size=8
python runner.py --config-name train_dl model=cnn_dn dataset=ChestMntdAcl hyp.batch_size=32
python runner.py --config-name train_dl model=cnn_dn dataset=Epilepsy hyp.batch_size=16
python runner.py --config-name train_dl model=cnn_dn dataset=WalkingSittingStanding hyp.batch_size=32


python runner.py --config-name train_dl model=attention task=imputation dataset=BasicMotions hyp.batch_size=8 hyp.learning_rate=1e-5
python runner.py --config-name train_dl model=attention dataset=BasicMotions hyp.batch_size=8 hyp.learning_rate=1e-5

python runner.py --config-name train_dl model=attention task=imputation dataset=ChestMntdAcl hyp.batch_size=32 hyp.learning_rate=1e-5
python runner.py --config-name train_dl model=attention dataset=ChestMntdAcl hyp.batch_size=32 hyp.learning_rate=1e-5


python runner.py --config-name train_dl model=attention task=imputation dataset=Epilepsy hyp.batch_size=16 hyp.learning_rate=1e-5
python runner.py --config-name train_dl model=attention dataset=Epilepsy hyp.batch_size=16 hyp.learning_rate=1e-5

python runner.py --config-name train_dl model=attention task=imputation dataset=WalkingSittingStanding hyp.batch_size=32 hyp.learning_rate=1e-5
python runner.py --config-name train_dl model=attention dataset=WalkingSittingStanding hyp.batch_size=32 hyp.learning_rate=1e-5

# TODO: Not tested yet
# python runner.py --config-name train_tsf model=tsf dataset=BasicMotions save=false
# python runner.py --config-name train_tsf model=tsf dataset=ChestMntdAcl save=false
# python runner.py --config-name train_tsf model=tsf dataset=Epilepsy save=false
# python runner.py --config-name train_tsf model=tsf dataset=WalkingSittingStanding save=false