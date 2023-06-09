for char in mvel1 mom1m idiovol retvol mom6m beta mom12m turn ill baspread 
# betasq mom36m std_turn dolvol zerotrade indmom maxret dy bm chmom nincr std_dolvol sp rd_sale roaq
do
    nohup python -u main.py --Model 'CA0 CA1 CA2 CA3' --K '5' --omit_char $char > logs/$char.log 2>&1
done
