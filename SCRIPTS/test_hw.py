def flatten_nested_list(nested) -> list:
    all_elements = []
    if nested:
        for element in nested:
            if isinstance(element,list):
                temp_elements = flatten_nested_list(element)
                for i in range(len(temp_elements)):
                    all_elements.append(temp_elements[i])
            else:
                all_elements.append(element)

    return all_elements