# Text-to-Math-Equation
(Part 2 ): As part of COL775 (Deep Learning), created an lstm-lstm-attention based architecture to convert text input into an equation format.

### Running instructions:

```python3 infer.py --model_file <path to the trained model> --beam_size [1 | 10 | 20 ]  --model_type [ lstm_lstm | lstm_lstm_attn | bert_lstm_attn_frozen | bert_lstm_attn_tuned] --test_data_file <the json file containing the problems>```

Example: 

```python3 infer.py -model_file models/lstm_lstm_attn.pth --model_type lstm_lstm_attn --test_data_file math_questions_test.json```


The script should write the output to the same JSON file with the extra field ‘predicted’  


(Part 1): Image Classification using ResNet built from scratch on Indian-Birds Dataset.
