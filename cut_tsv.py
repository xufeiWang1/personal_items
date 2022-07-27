#分割tsv文件
import re
mytsv='/home/xufei/mosei_wxf/mosei_wxf_fairseq/all_mosei_xufei.tsv'
traintxt = '/home/xufei/trainfilename.txt'
validtxt = '/home/xufei/validfilename.txt'
testtxt = '/home/xufei/testfilename.txt'
traintsv='/home/xufei/mosei_wxf/mosei_wxf_fairseq/train_xufei_debug2.tsv'
validtsv='/home/xufei/mosei_wxf/mosei_wxf_fairseq/valid_xufei_debug2.tsv'
testtsv='/home/xufei/mosei_wxf/mosei_wxf_fairseq/test_xufei_debug2.tsv'

def cut_tsv(tsv, txt, new_tsv):
    a = 0
    k = 0
    with open(txt, 'r', encoding='utf-8') as txtf:
        while True:
            line = txtf.readline()  # 读取第一行
            if line =='':
                break
            if k == 1:
                a = a+1
            k = 1
            tmp = re.split(r"/|\.", line)
            with open(tsv, 'r', encoding='utf-8') as tsvf:
                while True:
                    line2 = tsvf.readline()
                    if line2 == '':
                        break
                    tmp2 = re.split(r"\_", line2)
                    if tmp[2] == tmp2[0]:
                        k = 0
                        with open(new_tsv, 'a', encoding='utf-8') as ntf:
                            ntf.write(line2)
                    if tmp[2] == tmp2[0]+'_'+tmp2[1]:
                        k = 0
                        with open(new_tsv, 'a', encoding='utf-8') as ntf:
                            ntf.write(line2)
                        
cut_tsv(mytsv,validtxt,validtsv)
cut_tsv(mytsv, traintxt, traintsv)
cut_tsv(mytsv, testtxt, testtsv)

   
