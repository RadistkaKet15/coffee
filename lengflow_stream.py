from typing import List, Union, Generator, Iterator, Optional
import requests
from pydantic import BaseModel, Field
import json
import logging
import base64
import uuid

# Настраиваем логирование
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Pipeline:
    class Valves(BaseModel):
        LANGFLOW_URL: str = Field(default="http://192.168.10.20:7862/api/v1/run/c9587c74-b2b2-4e1d-b1ed-644a19f930c7", 
                                description="Langflow API URL")
        LANGFLOW_API_KEY: str = Field(default="sk-***", description="Langflow API Key")
        TIMEOUT: int = Field(default=300, description="Timeout for requests in seconds")

    def __init__(self):
        self.id = "langflow_stream"
        self.name = "Langflow"
        self.valves = self.Valves()

    def _prepare_payload_with_history_and_files(
        self, 
        user_message: str, 
        messages: List[dict], 
        body: dict
    ) -> dict:
        """Подготавливает payload с историей сообщений и файлами"""
        logger.info("Preparing payload with history and files")
        
        # Базовый payload
        payload = {
            "input_value": user_message,
            "input_type": "chat",
            "output_type": "chat", 
            "session_id": body.get("session_id", f"session-{uuid.uuid4().hex[:8]}"),
            "tweaks": {}
        }
        
        # Добавляем историю сообщений
        if len(messages) > 1:
            try:
                dialog_context = self._format_message_history(messages[:-1])
                payload["input_value"] = f"{dialog_context}\nuser: {user_message}"
                logger.debug(f"Added message history: {len(messages)-1} previous messages")
            except Exception as e:
                logger.error(f"Error formatting message history: {e}")
        
        # Добавляем параметры из body
        if "temperature" in body:
            payload["tweaks"]["temperature"] = body["temperature"]
            logger.debug(f"Set temperature: {body['temperature']}")
        
        # Обрабатываем файлы
        files = body.get("files", [])
        if files:
            try:
                file_context = self._process_files(files)
                if file_context:
                    payload["input_value"] = f"{file_context}\n\n{payload['input_value']}"
                    logger.info(f"Added {len(files)} file(s) to context")
            except Exception as e:
                logger.error(f"Error processing files: {e}")
        
        return payload

    def _format_message_history(self, messages: List[dict]) -> str:
        """Форматирует историю сообщений в текст"""
        formatted_messages = []
        for msg in messages:
            role = msg.get('role', 'unknown')
            content = msg.get('content', '')
            if content.strip():  # Пропускаем пустые сообщения
                formatted_messages.append(f"{role}: {content}")
        
        return "\n".join(formatted_messages)

    def _process_files(self, files: List[dict]) -> Optional[str]:
        """Обрабатывает файлы и возвращает их содержимое как текст"""
        file_contents = []
        
        for file in files:
            try:
                file_name = file.get('name', 'unknown')
                mime_type = file.get('mime_type', '')
                content = file.get('content', '')
                
                # Обрабатываем только текстовые файлы
                if content and mime_type.startswith(('text/', 'application/json')):
                    # Декодируем base64
                    decoded_content = base64.b64decode(content).decode('utf-8')
                    file_contents.append(f"Содержимое файла '{file_name}':\n{decoded_content}")
                    logger.debug(f"Processed file: {file_name}")
                    
            except Exception as e:
                logger.warning(f"Failed to process file {file.get('name', 'unknown')}: {e}")
                continue
        
        return "\n\n".join(file_contents) if file_contents else None

    def _process_stream_response(self, response: requests.Response) -> Generator[str, None, None]:
        """Обрабатывает потоковый ответ от Langflow"""
        logger.info("Starting stream processing")
        buffer = ""
        token_count = 0
        
        try:
            for line in response.iter_lines(decode_unicode=True):
                if line:
                    try:
                        line_text = line.strip()
                        if line_text:
                            buffer += line_text
                            
                            # Обрабатываем буфер для извлечения полных JSON объектов
                            while buffer:
                                json_obj, remaining_buffer = self._extract_json_from_buffer(buffer)
                                
                                if json_obj:
                                    # Обрабатываем JSON объект
                                    chunk = self._process_json_event(json_obj)
                                    if chunk:
                                        token_count += 1
                                        yield chunk
                                    
                                    buffer = remaining_buffer
                                else:
                                    # Не нашли полный JSON, ждем больше данных
                                    break
                                    
                    except UnicodeDecodeError as e:
                        logger.warning(f"Unicode decode error: {e}")
                        continue
                    except Exception as e:
                        logger.error(f"Error processing line: {e}")
                        continue
        
        except requests.exceptions.ChunkedEncodingError as e:
            logger.error(f"Chunked encoding error: {e}")
        except Exception as e:
            logger.error(f"Stream processing error: {e}")
        finally:
            logger.info(f"Stream processing completed. Tokens processed: {token_count}")

    def _extract_json_from_buffer(self, buffer: str) -> tuple[Optional[dict], str]:
        """Извлекает JSON объект из буфера"""
        try:
            # Простая проверка - ищем начало и конец JSON объекта
            start_idx = buffer.find('{')
            if start_idx == -1:
                return None, buffer
            
            # Пытаемся найти конец JSON объекта
            brace_count = 0
            in_string = False
            escape = False
            
            for i, char in enumerate(buffer[start_idx:], start_idx):
                if escape:
                    escape = False
                    continue
                
                if char == '\\':
                    escape = True
                    continue
                    
                if char == '"':
                    in_string = not in_string
                    continue
                    
                if not in_string:
                    if char == '{':
                        brace_count += 1
                    elif char == '}':
                        brace_count -= 1
                        if brace_count == 0:
                            # Нашли полный JSON объект
                            json_str = buffer[start_idx:i+1]
                            remaining_buffer = buffer[i+1:]
                            
                            try:
                                json_obj = json.loads(json_str)
                                return json_obj, remaining_buffer
                            except json.JSONDecodeError:
                                continue
            
            return None, buffer
            
        except Exception as e:
            logger.error(f"Error extracting JSON from buffer: {e}")
            return None, buffer

    def _process_json_event(self, event_data: dict) -> Optional[str]:
        """Обрабатывает JSON событие из потока"""
        try:
            event_type = event_data.get("event")
            data = event_data.get("data", {})
            
            if event_type == "token":
                chunk = data.get("chunk", "")
                if chunk:
                    logger.debug(f"Received token: {chunk}")
                    return chunk
            
            elif event_type == "add_message":
                logger.debug("Received add_message event")
            
            elif event_type == "end":
                logger.info("Received end event - stream completed")
            
            else:
                logger.debug(f"Received unknown event type: {event_type}")
                
        except Exception as e:
            logger.error(f"Error processing JSON event: {e}")
        
        return None

    def pipe(
        self, 
        user_message: str, 
        model_id: str, 
        messages: List[dict], 
        body: dict
    ) -> Union[str, Generator, Iterator]:
        
        logger.info(f"Processing message for model: {model_id}")
        logger.debug(f"User message: {user_message[:100]}...")
        logger.debug(f"Previous messages: {len(messages)}")
        logger.debug(f"Body keys: {list(body.keys())}")

        try:
            # Подготавливаем данные для Langflow с историей и файлами
            payload = self._prepare_payload_with_history_and_files(
                user_message, messages, body
            )
            
            headers = {
                'Content-Type': 'application/json',
                'x-api-key': self.valves.LANGFLOW_API_KEY
            }
            
            # Отправляем запрос к Langflow с stream=True
            logger.info(f"Sending request to Langflow: {self.valves.LANGFLOW_URL}")
            response = requests.post(
                f"{self.valves.LANGFLOW_URL}?stream=true",
                json=payload,
                headers=headers,
                stream=True,
                timeout=self.valves.TIMEOUT
            )
            
            response.raise_for_status()
            logger.info("Langflow request successful, starting stream processing")
            
            # Возвращаем генератор для обработки потока
            return self._process_stream_response(response)
            
        except requests.exceptions.RequestException as e:
            logger.error(f"Request error: {e}")
            return f"API Error: {str(e)}"
        except Exception as e:
            logger.error(f"Unexpected error in pipe: {e}", exc_info=True)
            return f"Error: {str(e)}"

    async def on_startup(self):
        """Вызывается при запуске пайплайна"""
        logger.info(f"Starting up {self.name} pipeline")

    async def on_shutdown(self):
        """Вызывается при остановке пайплайна"""
        logger.info(f"Shutting down {self.name} pipeline")
