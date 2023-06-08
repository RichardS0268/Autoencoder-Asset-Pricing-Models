import sys
import os

# # stock-level characteristics with index corresponding to original paper
# annual_chara = {
#     'absacc': 1, 'acc': 2, 'age': 4, 'agr': 5, 'bm': 9,
#     'bm_ia': 10, 'cashdebt': 12, 'cashpr': 13, 'cfp': 14, 'cfp_ia': 15,  
#     'chatoia': 16, 'chcsho': 17, 'chempia': 18, 'chinv': 19, 'chpmia': 21,
#     'convind': 24, 'currat': 25, 'depr': 26, 'divi': 27, 'divo': 28,
#     'dy': 30, 'egr': 32, 'ep': 33, 'gma': 34, 'grcapx': 35,
#     'grltnoa': 36, 'herf': 37, 'hire': 38, 'invest': 42, 'lev': 43,
#     'lgr': 44, 'mve_ia': 52, 'operprof': 54, 'orgcap': 55, 'pchcapx_ia': 56,
#     'pchcurrat': 57, 'pchdepr': 58, 'pchgm_pchsale': 59, 'pchquick': 60, 'pchsale_pchinvt': 61,
#     'pchsale_pchrect': 62, 'pchsale_pchxsga': 63, 'pchsaleinv': 64, 'pctacc': 65, 'ps': 67, 
#     'quick': 68, 'rd': 69, 'rd_mve': 70, 'rd_sale': 71, 'realestate': 72, 
#     'roic': 77, 'salecash': 79, 'saleinv': 80, 'salerec': 81, 'secured': 82, 
#     'securedind': 83, 'sgr': 84, 'sin': 85, 'sp': 86, 'tang': 91, 'tb': 92
# }

# quarter_chara = {
#     'aeavol': 3, 'cash': 11, 'chtx': 22, 'cinvest': 23,
#     'ear': 31, 'ms': 50, 'nincr': 53, 'roaq': 74,
#     'roavol': 75, 'roeq': 76, 'rsup': 78, 'stdacc': 89, 'stdcf': 90
# }

# month_chara = {
#     'baspread': 6, 'beta': 7, 'betasq': 8, 'chmom': 20,
#     'dolvol': 29, 'idiovol': 39, 'ill': 40, 'indmom': 41,
#     'maxret': 45, 'mom12m': 46, 'mom1m': 47, 'mom36m': 48,
#     'mom6m': 49, 'mvel1': 51, 'pricedelay': 66, 'retvol': 73,
#     'std_dolvol': 87, 'std_turn': 88, 'turn': 93, 'zerotrade': 94
# }

CHARAS_LIST = ['absacc','acc','age','agr','bm','bm_ia','cashdebt','cashpr','cfp','cfp_ia','chatoia','chcsho','chempia','chinv','chpmia','convind','currat','depr','divi','divo','dy','egr','ep','gma','grcapx','grltnoa','herf','hire','invest','lev','lgr','mve_ia','operprof','orgcap','pchcapx_ia','pchcurrat','pchdepr','pchgm_pchsale','pchquick','pchsale_pchinvt','pchsale_pchrect','pchsale_pchxsga','pchsaleinv','pctacc','ps','quick','rd','rd_mve','rd_sale','realestate','roic','salecash','saleinv','salerec','secured','securedind','sgr','sin','sp','tang','tb','aeavol','cash','chtx','cinvest','ear','ms','nincr','roaq','roavol','roeq','rsup','stdacc','stdcf','baspread','beta','betasq','chmom','dolvol','idiovol','ill','indmom','maxret','mom12m','mom1m','mom36m','mom6m','mvel1','pricedelay','retvol','std_dolvol','std_turn','turn','zerotrade']


# default learning rate of CA model
CA_DR = 0.5 # drop out rate
CA_LR = 0.001 # learning rate

# out of sample period
OOS_start = 19870101
OOS_end = 20161231



class HiddenPrints:
    def __init__(self, activated=True):
        self.activated = activated
        self.original_stdout = None

    def open(self):
        sys.stdout.close()
        sys.stdout = self.original_stdout

    def close(self):
        self.original_stdout = sys.stdout
        sys.stdout = open(os.devnull, 'w')

    def __enter__(self):
        if self.activated:
            self.close()

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.activated:
            self.open()



def git_push(message):
    os.system('git add results')
    os.system(f'git commit -m "no_dropout: {message}"')
    os.system('git push')