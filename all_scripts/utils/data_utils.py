def flatten_schema(schema):
    """Flatten a JSON schema object into a readable string."""
    if not schema or 'properties' not in schema:
        return ""
    
    parts = []
    for key, value in schema['properties'].items():
        type_str = value.get('type', 'unknown')
        desc = value.get('description', '')
        
        if type_str == 'array' and 'items' in value:
            items_type = value['items'].get('type', 'unknown')
            type_str = f"array of {items_type}"
        
        parts.append(f"{key} ({type_str}): {desc}")
    
    return "; ".join(parts)


def structure_tool(tool_name, tool, passage_prefix='passage: '):
    """Create a structured passage string from a tool dictionary using its name.
    
    Args:
        tool_name: The name of the tool
        tool: The tool dictionary containing description, arguments, results, etc.
        passage_prefix: Prefix to add to the passage (default: 'passage: ')
    
    Returns:
        A formatted string representation of the tool
    """
    description_expanded = tool.get('description_expanded', tool.get('description', ''))
    
    args_str = flatten_schema(tool.get('arguments', {}))
    
    results_str = flatten_schema(tool.get('results', {}))
    
    synthetic_questions = tool.get('synthetic_questions', [])
    questions_str = "; ".join(synthetic_questions) if synthetic_questions else ""
    
    parts = [
        f"Tool: {tool_name}",
        f"Description: {description_expanded}",
    ]
    
    if args_str:
        parts.append(f"Arguments: {args_str}")
    
    if results_str:
        parts.append(f"Results: {results_str}")
    
    if questions_str:
        parts.append(f"Example questions: {questions_str}")
    
    passage = " | ".join(parts)
    return passage_prefix + passage

