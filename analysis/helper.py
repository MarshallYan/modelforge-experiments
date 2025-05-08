import re

def extract_config(config_str, key):
    key_match = re.search(r'\b'+key+r'\b', config_str)

    if not key_match:
        raise KeyError("No such key in the config string!")
    key_begin_idx = key_match.start()

    key_end_idx = config_str.find(' ', key_begin_idx)
    extracted_string = config_str[key_begin_idx : key_end_idx]

    count = 0
    while count < 1000:
        count += 1

        # match parentheses 
        if extracted_string.count(']') - extracted_string.count('[') < 0 or extracted_string.count(')') - extracted_string.count('(') < 0:
            key_end_idx = config_str.find(' ', key_end_idx + 1)
            extracted_string = config_str[key_begin_idx : key_end_idx]

        # get rid of the last ','
        elif extracted_string[-1] == ',':
            extracted_string = extracted_string[:-1]

        # extract dict from the extracted string
        else:
            values = extracted_string[len(key) + 1 : ]
            tail_length = values.count(']') - values.count('[') + values.count(')') - values.count('(')
            values = values[:len(values) - tail_length]

            # convert to dict
            try:
                values = int(values)
            except ValueError:
                try:
                    values = float(values)
                except ValueError:
                    try:
                        values = eval(values)
                    except NameError:
                        pass
            result = {key: values}
            return result