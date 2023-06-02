# stock-level characteristics with index corresponding to original paper
annual_chara = {
    'absacc': 1, 'acc': 2, 'age': 4, 'agr': 5, 'bm': 9,
    'bm_ia': 10, 'cashdebt': 12, 'cashpr': 13, 'cfp': 14, 'cfp_ia': 15,  
    'chatoia': 16, 'chcsho': 17, 'chempia': 18, 'chinv': 19, 'chpmia': 21,
    'convind': 24, 'currat': 25, 'depr': 26, 'divi': 27, 'divo': 28,
    'dy': 30, 'egr': 32, 'ep': 33, 'gma': 34, 'grcapx': 35,
    'grltnoa': 36, 'herf': 37, 'hire': 38, 'invest': 42, 'lev': 43,
    'lgr': 44, 'mve_ia': 52, 'operprof': 54, 'orgcap': 55, 'pchcapx_ia': 56,
    'pchcurrat': 57, 'pchdepr': 58, 'pchgm_pchsale': 59, 'pchquick': 60, 'pchsale_pchinvt': 61,
    'pchsale_pchrect': 62, 'pchsale_pchxsga': 63, 'pchsaleinv': 64, 'pctacc': 65, 'ps': 67, 
    'quick': 68, 'rd': 69, 'rd_mve': 70, 'rd_sale': 71, 'realestate': 72, 
    'roic': 77, 'salecash': 79, 'saleinv': 80, 'salerec': 81, 'secured': 82, 
    'securedind': 83, 'sgr': 84, 'sin': 85, 'sp': 86, 'tang': 91, 'tb': 92
}

quarter_chara = {
    'aeavol': 3, 'cash': 11, 'chtx': 22, 'cinvest': 23,
    'ear': 31, 'ms': 50, 'nincr': 53, 'roaq': 74,
    'roavol': 75, 'roeq': 76, 'rsup': 78, 'stdacc': 89, 'stdcf': 90
}

month_chara = {
    'baspread': 6, 'beta': 7, 'betasq': 8, 'chmom': 20,
    'dolvol': 29, 'idiovol': 39, 'ill': 40, 'indmom': 41,
    'maxret': 45, 'mom12m': 46, 'mom1m': 47, 'mom36m': 48,
    'mom6m': 49, 'mvel1': 51, 'pricedelay': 66, 'retvol': 73,
    'std_dolvol': 87, 'std_turn': 88, 'turn': 93, 'zerotrade': 94
}

charas = list(annual_chara.keys()) + list(quarter_chara.keys()) + list(month_chara.keys())