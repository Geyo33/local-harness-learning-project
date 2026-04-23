def add(a: float, b: float) -> str:
    """Add two numbers and return the result."""
    return str(a + b)


def subtract(a: float, b: float) -> str:
    """Subtract b from a and return the result."""
    return str(a - b)


def multiply(a: float, b: float) -> str:
    """Multiply two numbers and return the result."""
    return str(a * b)


def divide(a: float, b: float) -> str:
    """Divide a by b and return the result. Returns an error if b is zero."""
    if b == 0:
        return "Error: division by zero"
    return str(a / b)


DISPATCH = {
    "add": add,
    "subtract": subtract,
    "multiply": multiply,
    "divide": divide,
}
