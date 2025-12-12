import logging
from typing import Optional

import chainlit as cl


class ErrorHandler:
    """Centralized error handling for the application."""
    
    # Error message templates
    ERROR_MESSAGES = {
        ValueError: {
            "prefix": "⚠️",
            "user_message": "Validation error: {error}",
            "log_level": logging.WARNING,
        },
        ConnectionError: {
            "prefix": "❌",
            "user_message": "Connection error. Please check if the services are running and try again.",
            "log_level": logging.ERROR,
        },
        Exception: {
            "prefix": "❌",
            "user_message": "An unexpected error occurred. Please try again or contact support if the issue persists.",
            "log_level": logging.ERROR,
        },
    }
    
    @classmethod
    async def handle_error(
        cls,
        error: Exception,
        context: str,
        show_details: bool = False
    ) -> None:
        """Handle and display errors to the user.
        
        Args:
            error: The exception that occurred
            context: Context description (e.g., "processing files", "generating response")
            show_details: Whether to show error details to user (default False for security)
        """
        error_type = type(error)
        
        # Get error config, default to generic Exception config
        config = cls.ERROR_MESSAGES.get(error_type, cls.ERROR_MESSAGES[Exception])
        
        # Log the error
        log_message = f"Error during {context}: {str(error)}"
        if config["log_level"] == logging.ERROR:
            logging.error(log_message, exc_info=True)
        else:
            logging.log(config["log_level"], log_message)
        
        # Prepare user message
        if show_details and isinstance(error, ValueError):
            # For validation errors, show the actual error message
            user_message = f"{config['prefix']} {str(error)}"
        else:
            user_message = f"{config['prefix']} {config['user_message'].format(error=str(error))}"
        
        # Send message to user
        await cl.Message(content=user_message).send()
    
    @classmethod
    def get_error_message(cls, error: Exception, show_details: bool = False) -> str:
        """Get formatted error message without sending it.
        
        Args:
            error: The exception that occurred
            show_details: Whether to include error details
            
        Returns:
            Formatted error message string
        """
        error_type = type(error)
        config = cls.ERROR_MESSAGES.get(error_type, cls.ERROR_MESSAGES[Exception])
        
        if show_details and isinstance(error, ValueError):
            return f"{config['prefix']} {str(error)}"
        else:
            return f"{config['prefix']} {config['user_message'].format(error=str(error))}"
