#!/bin/bash
if [ $ACTION = "TRAIN" ]
then
	python3 PrepareDataset.py
	python3 TrainModel.py
	python3 EvaluateModel.py

	if [ $? -eq 2 ]
	then
		python3 TrainModel_Outliers.py
		python3 EvaluateModel.py
	fi

	if [ $? -eq 0 ]
	then
		echo "Training successful, deploying model"
		cp -f model/AE.pt model/AE_prod.pt
	else
		echo "Training failed, keeping existing model"
	fi
else
	thr=$(cat model/threshold.dat)
	if [ "$HOOK_TEAMS" = "" ]
	then
		python3 Monitoring.py -t $thr &
	else
		python3 Monitoring.py -t $thr -h $HOOK_TEAMS &
	fi
	python3 MonitoringService.py
fi
