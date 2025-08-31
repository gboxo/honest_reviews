def reward_func(completions, **kwargs):  # change this out for other more complicated ones later.
    """Reward function that rewards completions with more unique letters."""
    completion_contents = [completion[0]["content"] for completion in completions]
    return [float(len(set(content))) for content in completion_contents]



"""
later we will need to parse together the code contests thing and see how they work together."""
