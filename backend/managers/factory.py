from backend.managers.knowledge_manager import KnowledgeManager
from backend.managers.chat_manager import ChatManager


def create_app_manager(load_data: bool = True) -> ChatManager:
    """
    Creates and bootstraps the KnowledgeManager and ChatManager.
    Returns a ready-to-use ChatManager instance.

    Args:
        load_data: if True, sync FAQ/operator knowledge into the vector DB.
                  For evaluation runs against an already-prepared DB, set to False.
    """
    knowledge_manager = KnowledgeManager()

    if load_data:
        knowledge_manager.load_faq_data()
        knowledge_manager.load_operator_knowledge()

    return ChatManager(knowledge_manager=knowledge_manager)
