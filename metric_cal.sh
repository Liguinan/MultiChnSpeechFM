python tools/calculate_metrics.py \
  -R /path/to/MultiChnSpeechFMExperiments_TCN/TAC_Based_MultiChnNet/inference/array_0_2_4/reference/wav.scp \
  -E /path/to/MultiChnSpeechFMExperiments_TCN/TAC_Based_MultiChnNet/inference/array_0_2_4/noisy/wav.scp \
  -M SI_SDR,STOI,WB_PESQ,NB_PESQ