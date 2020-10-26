#PBS -N test_pytorch
#PBS -l nodes=1:ppn=2:gpus=1
#PBS -l walltime=00:01:00
#PBS -j oe
#PBS -A PHS0336 

# submit with 'qsub [name of the script]'
# find out start time with 'showstart [job id]'
# find stats of the queue with 'qstat -u medirz90'
# delete job with 'qdel [job id]'
# or with 'qstop [job id]'  
# see the balance with 'OSCusage'

set -x

module load cuda/10.1.168
cd $HOME
source .bashrc

cd $HOME/github/sysnetdev/scripts/
source activate sysnet

# manually add the path, later we will install the pipeline with `pip`
export PYTHONPATH=${HOME}/github/sysnetdev:${PYTHONPATH}

python app.py -ax {0..17} 
