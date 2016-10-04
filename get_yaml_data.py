import yaml
import shelve
from pylab import array

# 1. Get yaml data
# open yaml file
ifile = open('neha/data/neha_data_revised.yml');
# read in data string
filestr = ifile.read();
# close file
ifile.close();
# parse data string into object (list of dicts)
neha_data = yaml.load(filestr);


total_cases = len(neha_data);

# 2. Arrange the data in a format similar to that used for the original analysis
# i.e., in arrays with two columns representing rt and confidence
rem_hit = array([(el['rt.normed'],el['confidence']) for el in neha_data \
                if el['judgment']=='hit' and el['rk.response']=='remember']);
know_hit = array([(el['rt.normed'],el['confidence']) for el in neha_data \
                if el['judgment']=='hit' and el['rk.response']=='know']);
rem_fa = array([(el['rt.normed'],el['confidence']) for el in neha_data \
                if el['judgment']=='FA' and el['rk.response']=='remember']);
know_fa = array([(el['rt.normed'],el['confidence']) for el in neha_data \
                if el['judgment']=='FA' and el['rk.response']=='know']);
CR = array([(el['rt.normed'],el['confidence']) for el in neha_data \
                if el['judgment']=='CR']);
miss = array([(el['rt.normed'],el['confidence']) for el in neha_data \
                if el['judgment']=='miss']);

# 3. Save the reformatted data into a shelve database
db = shelve.open('neha/data/neha_data.dat','n');
db['rem_hit'] = rem_hit;
db['know_hit'] = know_hit;
db['rem_fa'] = rem_fa;
db['know_fa'] = know_fa;
db['CR'] = CR;
db['miss'] = miss;
db.close();