def format_collection_name(original_name):
    """
    Converts a filename to a ChromaDB-compatible format.
    """
    # Remove file extension if present
    if '.' in original_name:
        name_parts = original_name.split('.')
        base_name = '.'.join(name_parts[:-1])  # Join everything except the last part
    else:
        base_name = original_name
    
    # Replace spaces with underscores
    formatted_name = base_name.replace(' ', '_')
    
    # Replace colons with hyphens
    formatted_name = formatted_name.replace(':', '-')
    
    # Ensure the name starts and ends with alphanumeric characters
    # by trimming any non-alphanumeric characters from the start and end
    while formatted_name and not formatted_name[0].isalnum():
        formatted_name = formatted_name[1:]
    
    while formatted_name and not formatted_name[-1].isalnum():
        formatted_name = formatted_name[:-1]
    
    # If the name is too short after processing, add a default prefix
    if len(formatted_name) < 3:
        formatted_name = "collection_" + formatted_name
    
    return formatted_name