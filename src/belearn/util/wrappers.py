def static_state_decorator(func):
    """Decorator that stops the function from changing the state

    Args:
        func (method): any method
    """
    def wrapper(*args, **kwargs):

        # saves the current state
        current_state = args[0].get_state

        # runs the function
        out = func(*args, **kwargs)

        # resets the state
        args[0].set_attributes(**current_state)

        # returns the output
        return out

    # returns the wrapper
    return wrapper