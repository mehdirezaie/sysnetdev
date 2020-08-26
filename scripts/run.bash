#----
# cont mock 001 with L1
#----


find_lr=false  # find learning rate
find_nhl=false  # find num of hidden layers
find_l1=false    # find L1 scale
run_nn=true


# hyper-parameters
lr=0.03
batchsize=512
optim=adamw
nhl='5 20'
nepoch=150
l1=1.0e-5

# path to input and output
data_dir=../input/001/cp2p/cp2p_001.hp.256.5.r.npy
data_out=../output/mock001_cp2p_${optim} 



if [ "${find_lr}" = true ];then
    echo "run LR finder ..."
    python app.py -i ${data_dir} -o ${data_out}/find_lr --optim ${optim} \
        -bs ${batchsize} --model dnn --loss mse --ax {0..17} -fl
fi

if [ "${find_nhl}" = true ];then
    echo "run NN num of hidden layer finder ..."
    python app.py -i ${data_dir} -o ${data_out}/find_nhl --optim ${optim} \
        -bs ${batchsize} --model dnn --loss mse -ax {0..17} -lr ${lr} -fs
fi

if [ "${find_l1}" = true ];then
    echo "run L1 finder ..."
    python app.py -i ${data_dir} -o ${data_out}/find_l1 --optim ${optim} \
        -bs ${batchsize} --model dnn --loss mse -ax {0..17} -lr ${lr} \
        --nn_structure ${nhl} -fl1
fi

if [ "${run_nn}" = true ];then
    echo "run NN with ${lr} ${batchsize} ${optim} ${nhl}  ..."
    python app.py -i ${data_dir} -o ${data_out}/model --optim ${optim} \
        -bs ${batchsize} --model dnn --loss mse -ax {0..17} -lr ${lr} \
        --nn_structure ${nhl} -ne ${nepoch} --l1_alpha ${l1} -k


    python app.py -i ${data_dir} -o ${data_out}/model_wol1 --optim ${optim} \
        -bs ${batchsize} --model dnn --loss mse -ax {0..17} -lr ${lr} \
        --nn_structure ${nhl} -ne ${nepoch} --l1_alpha -1.0 -k
fi

#python app.py -ax {0..17} -i ../input/001/cp2p/cp2p_001.hp.256.5.r.npy -o ../output/mock001_cp2p_l1_1pm3 -ne 100 -lr 0.01 --l1_alpha 0.001
#python app.py -ax {0..17} -i ../input/001/cp2p/cp2p_001.hp.256.5.r.npy -o ../output/mock001_cp2p_l1_1pm2 -ne 100 -lr 0.01 --l1_alpha 0.01
#python app.py -ax {0..17} -i ../input/001/cp2p/cp2p_001.hp.256.5.r.npy -o ../output/mock001_cp2p_l1_1pm1 -ne 100 -lr 0.01 --l1_alpha 0.1
