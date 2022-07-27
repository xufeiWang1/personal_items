python examples/speech_recognition_sjtu/prep_data_npyfile.py --output-root /mnt/xlancefs/home/xc095/gigaspeech-dev/data4fairseq -w /mnt/xlancefs/home/xc095/gigaspeech-dev/gigaspeech-dev.info --prefix dev-tidy

# data pre-processing
python modify_df.py ~/data/cmu_mosei/data4fariseq/train.tsv ~/data/cmu_mosei/data4fariseq/train_mod.tsv ~/data/cmu_mosei/data4fariseq/vocab.txt
python modify_df.py ~/data/cmu_mosei/data4fariseq/test.tsv ~/data/cmu_mosei/data4fariseq/test_mod.tsv
python modify_df.py ~/data/cmu_mosei/data4fariseq/valid.tsv ~/data/cmu_mosei/data4fariseq/valid_mod.tsv

# install fairseq
pip install --editable ./
python setup.py build_ext --inplace

# Train models
python fairseq_cli/hydra_train.py --config-dir examples/speech_recognition_sjtu/mosei/ --config-name config-ctc
python fairseq_cli/hydra_train.py --config-dir examples/emotion_recognition/mosei/ --config-name config-transformer

git push -u origin emotion_recognition
