from contextlib import contextmanager
from functools import wraps

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




@contextmanager
def temporary_state(obj, **modifications):
    # Create a deep copy of the object's state
    original_state = obj.get_state.copy()
    try:
        # Apply modifications to the object
        obj.set_attributes(**modifications)
        yield obj
    finally:
        # Restore the original state
        obj.set_attributes(**original_state)


def context_manager_decorator(func):
    """Decorator that wraps a function inside a temporary state context."""
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        with temporary_state(self,**kwargs.get('true', {})):  # Use the true state if provided
            return func(self, *args, **kwargs)  # Call the function with modified state
    return wrapper