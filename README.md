# Autoencoder-Asset-Pricing-Models

[notion link](https://richard0268.notion.site/Project-2-9c4da14827f84ba7bbe77c67d9169f6f)

```bash
# generate preprocessed data and download portfolio returns
python data_prepare.py

# 
python main.py --Model 'CA0 CA1 CA2 CA3' --K '1 2 3 4 5 6'

nohup python -u main.py --Model 'CA0' --K '5' --omit_char 'mvel1 mom1m idiovol retvol mom6m beta mom12m turn ill baspread betasq mom36m std_turn dolvol zerotrade' > logs/CA0.log 2>&1 &
```
