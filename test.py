import re 
def state_check(str):
    st_short = ['AK', 'AR', 'AZ', 'CA', 'CO', 'CT', 'DC', 'DE', 'FL', 'GA', \
                'HA', 'IA', 'ID', 'IL', 'IN', 'KS', 'KY', 'LA', 'MA', 'MD', \
                'ME', 'MI', 'MN', 'MO', 'MS', 'MT', 'NC', 'ND', 'NE', 'NH', \
                'NJ', 'NM', 'NV', 'NY', 'OH', 'OK', 'OR', 'PA', 'PR', 'RI', \
                'SC', 'SD', 'TN', 'TX', 'UT', 'VA', 'VT', 'WA', 'WI', 'WV', \
                'WY', 'N.Y.', 'N.Y', 'D.C.', 'D.C']
    st_long = ['Alaska', 'Arkansas', 'Arizona', 'California', 'Colorado', \
               'Connecticut', 'District of Columbia', 'Delaware', \
               'Florida', 'Georgia', 'Hawaii', 'Iowa', 'Idaho', 'Illinois', \
               'Indiana', 'Kansas', 'Kentucky', 'Louisiana', 'Massachusetts', \
               'Maryland', 'Maine', 'Michigan', 'Minnesota', 'Missouri', \
               'Mississippi', 'Montana', 'North Carolina', 'North Dakota', \
               'Nebraska', 'New Hampshire', 'New Jersey', 'New Mexico', \
               'Nevada', 'New York', 'Ohio', 'Oklahoma', 'Oregon','Pennsylvania', \
               'Puerto Rico', 'Rhode Island', 'South Carolina', 'South Dakota', \
               'Tennessee', 'Texas', 'Utah', 'Virginia', 'Vermont', 'Washington', \
               'Wisconsin', 'West Virginia', 'Wyoming']

    #st_short = [st.lower() for st in st_short]
    #st_long = [st.lower() for st in st_long]
    #str = str.lower()
    str = re.sub(r'[^a-zA-Z0-9.,]', ' ', str)
    wrds = str.split()
    for i, wrd in enumerate(wrds[:-1]):
        if wrd in ['New'] and wrds[i+1] in ['Hampshire', 'Jersey', 'Mexico', 'York']:
            add_state = wrd + ' ' + wrds[i+1]
            return add_state
        else:
            if wrd in ['North'] and wrds[i+1] in ['Carolina', 'Dakota']:
                add_state = wrd + ' ' + wrds[i+1]
                return add_state
            else:
                if wrd in ['South'] and wrds[i+1] in ['Carolina', 'Dakota']:
                    add_state = wrd + ' ' + wrds[i+1]
                    return add_state
                else:
                    if wrd in ['West'] and wrds[i+1] in ['Virginia']:
                        add_state = wrd + ' ' + wrds[i+1]
                        return add_state
                    else:
                        if wrd in ['Rhode'] and wrds[i+1] in ['Island']:
                            add_state = wrd + ' ' + wrds[i+1]
                            return add_state
    if ['district'] in wrds:
        idx = wrds.index('district')
        if idx < len(wrds) - 2 and \
           wrds[idx+1] == 'of' and wrds[idx+2] == 'columbia':
            add_state = 'District of Columbia'
            return add_state
                    
    add_state = []
    for wrd in wrds :
        if wrd in st_short or wrd in st_long:
            add_state.append(wrd)
            return add_state[0]
        
    #if add_state[0] != '':
    #    return str
    #else:
    return

str="Louisiana"

print(state_check(str))
        
