from datetime import datetime
from ultralytics import YOLO
import matplotlib.pyplot as plt
import requests
import logging
import base64
import time
import json
import cv2

plt.style.use('dark_background')

class OverColetor:
    def __init__(self, empresa_id, empresa_nome, coletor_id, coletor_descricao, coletor_localizacao,
                 camera_source=0, model_key="oil_spill", model_path="models/Vo1.pt", show_notebook=True,
                 capture_interval=15, confidence_threshold=0.7, ngrok_domain=""):
        """Inicializa o detector de manchas de óleo - versão corrigida"""
        
        # Configurações básicas
        self.EMPRESA_ID = empresa_id
        self.EMPRESA_NOME = empresa_nome
        self.COLETOR_ID = coletor_id
        self.MODELO_ID = model_key
        self.COLETOR_DESCRICAO = coletor_descricao
        self.COLETOR_LOCALIZACAO = coletor_localizacao
        
        # Configurações de operação
        self.CAPTURE_INTERVAL_SECONDS = capture_interval
        self.CONFIDENCE_THRESHOLD = confidence_threshold
        self.CAMERA_SOURCE = camera_source
        self.SHOW_NOTEBOOK = show_notebook
        
        # URLs do servidor - CORRIGIDO
        base_url = f"https://{ngrok_domain}.ngrok-free.app"
        self.OIL_DETECTION_URL = f"{base_url}/oil_detection"  # Endpoint correto
        self.SKF_ANALYSIS_URL = f"{base_url}/skf_analysis"
        self.HEALTH_URL = f"{base_url}/health"
        
        # Caminho do modelo
        self.MODEL_PATH = model_path
        
        # Configurar log básico
        self._setup_logger()
        
        # Testar conexão na inicialização
        self._test_server_connection()
        
        # Inicializar câmera uma vez só para manter conexão ativa
        self._init_camera()
        
    def _setup_logger(self):
        """Configura log básico"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger('OverColetor')
    
    def _test_server_connection(self):
        """Testa conexão com o servidor"""
        try:
            response = requests.get(self.HEALTH_URL, timeout=10)
            if response.ok:
                health_data = response.json()
                self.logger.info(f"✓ Servidor conectado: {health_data.get('status', 'unknown')}")
                return True
            else:
                self.logger.error(f"✗ Servidor não respondeu adequadamente: {response.status_code}")
                return False
        except Exception as e:
            self.logger.error(f"✗ Erro ao conectar com servidor: {str(e)}")
            return False
    
    def _init_camera(self):
        """Inicializa a câmera e mantém conexão ativa"""
        try:
            self.cap = cv2.VideoCapture(self.CAMERA_SOURCE)
            if not self.cap.isOpened():
                raise ValueError(f"Não foi possível abrir câmera {self.CAMERA_SOURCE}")
            
            # Configurar resolução máxima da câmera
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
            
            # Configurar buffer da câmera para 1 frame (mais recente)
            self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            
            # Configurar FPS (se suportado)
            self.cap.set(cv2.CAP_PROP_FPS, 30)
            
            self.logger.info("✓ Câmera inicializada e configurada")
            
        except Exception as e:
            self.logger.error(f"✗ Erro ao inicializar câmera: {str(e)}")
            self.cap = None
    
    def _capture_fresh_frame(self):
        """Captura um frame fresco, descartando frames antigos do buffer"""
        if self.cap is None or not self.cap.isOpened():
            raise ValueError("Câmera não está inicializada ou disponível")
        
        # Descartar frames antigos do buffer lendo múltiplos frames
        # Isso garante que pegamos o frame mais recente
        for _ in range(5):  # Lê até 5 frames para limpar o buffer
            ret, frame = self.cap.read()
            if not ret:
                break
        
        if not ret:
            raise ValueError("Não foi possível capturar frame da câmera")
        
        return frame
    
    def _encode_image_high_quality(self, image):
        """Codifica imagem em base64 mantendo alta qualidade"""
        try:
            # Usar PNG com compressão mínima para manter qualidade
            encode_params = [cv2.IMWRITE_PNG_COMPRESSION, 1]  # Compressão mínima (0-9, onde 0=sem compressão)
            success, buffer = cv2.imencode('.png', image, encode_params)
            
            if not success:
                raise ValueError("Falha ao codificar imagem")
            
            # Converter para base64
            image_b64 = base64.b64encode(buffer).decode('utf-8')
            
            # Log do tamanho para debug
            size_mb = len(image_b64) * 3/4 / (1024*1024)  # Aproximação do tamanho em MB
            #self.logger.debug(f"Imagem codificada: {size_mb:.2f}MB")
            
            return image_b64
            
        except Exception as e:
            self.logger.error(f"Erro ao codificar imagem: {str(e)}")
            return None
    
    def _send_data_to_server(self, original_image, processed_image, detection_data, metadata):
        """Envia dados para o servidor usando o endpoint correto"""
        try:
            # Codificar imagens em base64 com alta qualidade
            #self.logger.info("Codificando imagens...")
            imagem_original_b64 = self._encode_image_high_quality(original_image)
            imagem_processada_b64 = self._encode_image_high_quality(processed_image)
            
            if not imagem_original_b64 or not imagem_processada_b64:
                self.logger.error("Falha na codificação das imagens")
                return False
            
            # Criar timestamp ISO format
            now = datetime.now()
            timestamp_iso = now.isoformat()
            timestamp_file = now.strftime('%Y%m%d_%H%M%S_%f')[:-3]  # Formato para nome do arquivo
            detections_count = len(detection_data)
            
            # Nomes dos arquivos
            original_image_name = f"RAW-{self.EMPRESA_ID}-{self.COLETOR_ID}-{self.MODELO_ID}-{timestamp_file}-{detections_count}.png"
            processed_image_name = f"PRC-{self.EMPRESA_ID}-{self.COLETOR_ID}-{self.MODELO_ID}-{timestamp_file}-{detections_count}.png"
            
            # Preparar payload conforme esperado pelo servidor
            payload = {
                "imagem_original_base64": imagem_original_b64,
                "imagem_processada_base64": imagem_processada_b64,
                "name_imagem_original": original_image_name,
                "name_imagem_processada": processed_image_name,
                "empresa_info": {
                    "empresa_id": self.EMPRESA_ID,
                    "empresa_nome": self.EMPRESA_NOME,
                    "coletor_id": self.COLETOR_ID,
                    "modelo_id": self.MODELO_ID,
                    "coletor_descricao": self.COLETOR_DESCRICAO,
                    "localizacao": self.COLETOR_LOCALIZACAO
                },
                "detection_data": detection_data,
                "metadata": metadata,
                "timestamp": timestamp_iso,  # ISO format para timestamp
                "confidence_threshold": max([d['confidence'] for d in detection_data]) if detection_data else self.CONFIDENCE_THRESHOLD,
                "detections_count": detections_count
            }
            
            # Log do tamanho do payload
            payload_size_mb = len(json.dumps(payload).encode('utf-8')) / (1024*1024)
            self.logger.info(f"Enviando payload de {payload_size_mb:.2f}MB com {detections_count} detecções...")
            
            # Headers adequados
            headers = {
                'Content-Type': 'application/json',
                'Accept': 'application/json'
            }
            
            # Enviar dados com timeout maior para imagens grandes
            response = requests.post(
                self.OIL_DETECTION_URL, 
                json=payload, 
                headers=headers,
                timeout=60  # Timeout maior para imagens de alta qualidade
            )
            
            # Log da resposta completa para debug
            self.logger.info(f"Status da resposta: {response.status_code}")
            
            if response.ok:
                try:
                    result = response.json()
                    self.logger.debug(f"Resposta do servidor: {result}")
                    
                    # Verificar se foi processado com sucesso
                    if result.get("status") == "success" or result.get("return") == 1:
                        deteccao_id = result.get("deteccao_id")
                        self.logger.info(f"✓ Captura & Dados Enviados :: ID {deteccao_id}")
                        return True
                    else:
                        self.logger.warning(f"✗ Servidor não confirmou sucesso: {result}")
                        return False
                        
                except json.JSONDecodeError:
                    self.logger.error(f"✗ Resposta inválida do servidor: {response.text}")
                    return False
            else:
                self.logger.error(f"✗ Erro HTTP {response.status_code}: {response.text}")
                return False
                
        except requests.exceptions.Timeout:
            self.logger.error("✗ Timeout ao enviar dados (imagem muito grande?)")
            return False
        except requests.exceptions.ConnectionError:
            self.logger.error("✗ Erro de conexão com servidor")
            return False
        except Exception as e:
            self.logger.error(f"✗ Erro ao enviar dados: {str(e)}")
            return False
    
    def _detect_and_send(self):
        """Captura imagem FRESCA, executa detecção e envia dados"""
        try:
            # CAPTURAR IMAGEM FRESCA NO MOMENTO EXATO
            self.logger.debug("Capturando câmera...")
            original_image = self._capture_fresh_frame()
            capture_timestamp = datetime.now()  # Timestamp exato da captura
            
            #self.logger.info(f"Imagem capturada: {original_image.shape[1]}x{original_image.shape[0]} às {capture_timestamp.strftime('%H:%M:%S.%f')[:-3]}")
            
            # Carregar modelo (só depois da captura)
            #self.logger.debug("Carregando modelo YOLO...")
            model = YOLO(self.MODEL_PATH, verbose=False)
            
            # Executar detecção (só depois da captura)
            #self.logger.debug("Executando detecção YOLO...")
            results = model(original_image, conf=self.CONFIDENCE_THRESHOLD, verbose=False)
            #import yaml
            #with open('results.yaml', 'w') as f:
            #    yaml.dump(results, f)
            # Processar resultados
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': result.names[int(box.cls)],
                        'confidence': round(float(box.conf), 6),  # Mais precisão
                        'bbox': [round(coord, 2) for coord in box.xyxy[0].tolist()],
                        'bbox_normalized': [round(coord, 6) for coord in box.xywhn[0].tolist()]  # Coordenadas normalizadas
                    })
            
            #self.logger.info(f"Detecções encontradas: {len(detections)}")
            
            # Se não há detecções, apenas log (não enviar)
            if len(detections) == 0:
                #self.logger.info("Nenhuma detecção encontrada - não enviando")
                return False
            
            processed_image = results[0].plot()

            # Mostrar imagens se solicitado
            if self.SHOW_NOTEBOOK:
                self._display_results(original_image, processed_image, detections)
            
            # Criar metadados mais detalhados (incluindo timestamp exato da captura)
            metadata = {
                'model_info': {
                    'model_path': self.MODEL_PATH,
                    'confidence_threshold': self.CONFIDENCE_THRESHOLD
                },
                'image_info': {
                    'original_shape': original_image.shape,
                    'channels': original_image.shape[2] if len(original_image.shape) > 2 else 1
                },
                'inference_speed': {
                    'preprocess_ms': round(results[0].speed['preprocess'], 4),
                    'inference_ms': round(results[0].speed['inference'], 4),
                    'postprocess_ms': round(results[0].speed['postprocess'], 4)
                },
                'detection_summary': {
                    'total_detections': len(detections),
                    'max_confidence': max([d['confidence'] for d in detections]) if detections else 0,
                    'classes_detected': list(set([d['class'] for d in detections]))
                },
                'timestamp_capture': capture_timestamp.isoformat(),  # Timestamp exato da captura
                'timestamp_processing': datetime.now().isoformat()   # Timestamp do processamento
            }
            
            # Enviar dados
            send_success = self._send_data_to_server(original_image, processed_image, detections, metadata)
            
            return send_success
            
        except Exception as e:
            self.logger.error(f"Erro na detecção: {str(e)}")
            return False
    
    def _display_results(self, original_image, processed_image, detections):
        """Exibe resultados da detecção"""
        try:
            original_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            plt.figure(figsize=(15, 6))
            
            plt.subplot(1, 2, 1)
            plt.imshow(original_rgb)
            plt.title('Imagem Original')
            plt.axis('off')
            
            plt.subplot(1, 2, 2)
            plt.imshow(processed_rgb)
            plt.title(f'Detecções ({len(detections)} encontradas)')
            plt.axis('off')
            
            # Adicionar lista de detecções
            detection_text = "\n".join([
                f"{d['class']}: {d['confidence']:.3f}" for d in detections[:5]  # Mostrar apenas 5 primeiras
            ])
            if len(detections) > 5:
                detection_text += f"\n... e mais {len(detections)-5}"
            
            plt.figtext(0.02, 0.02, detection_text, fontsize=8, verticalalignment='bottom')
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            self.logger.warning(f"Erro ao exibir resultados: {str(e)}")
    
    def test_single_detection(self):
        """Executa uma única detecção para teste"""
        self.logger.info("=== TESTE DE DETECÇÃO ÚNICA ===")
        success = self._detect_and_send()
        if success:
            self.logger.info("✓ Teste concluído com sucesso!")
        else:
            self.logger.warning("✗ Teste falhou!")
        return success
    
    def run_continuous_monitoring(self):
        """Executa monitoramento contínuo com captura instantânea"""
        #self.logger.info("=== INICIANDO MONITORAMENTO CONTÍNUO ===")
        self.logger.info(f"Empresa: {self.EMPRESA_NOME} ({self.EMPRESA_ID})")
        self.logger.info(f"Coletor: {self.COLETOR_DESCRICAO} ({self.COLETOR_ID})")
        self.logger.info(f"Intervalo: {self.CAPTURE_INTERVAL_SECONDS}s")
        self.logger.info(f"Confiança: {self.CONFIDENCE_THRESHOLD}")
        self.logger.info("✓ Captura instantânea ativa - sempre pega frame mais recente")
        
        successful_sends = 0
        total_attempts = 0
        
        try:
            while True:
                cycle_start = time.time()
                total_attempts += 1
                #self.logger.info(f"\n--- Captura #{total_attempts} ---")
                
                success = self._detect_and_send()
                if success:
                    successful_sends += 1
                
                success_rate = (successful_sends / total_attempts) * 100
                cycle_duration = time.time() - cycle_start
                
                #self.logger.info(f"Taxa de sucesso: {successful_sends}/{total_attempts} ({success_rate:.1f}%)")
                #self.logger.info(f"Duração do ciclo: {cycle_duration:.2f}s")
                
                # Aguardar próxima captura (considerando o tempo já gasto)
                if self.CAPTURE_INTERVAL_SECONDS > 0:
                    wait_time = max(0, self.CAPTURE_INTERVAL_SECONDS - cycle_duration)
                    if wait_time > 0:
                        #self.logger.info(f"Aguardando {wait_time:.2f}s...")
                        time.sleep(wait_time)
                
        except KeyboardInterrupt:
            #self.logger.info(f"\n=== MONITORAMENTO INTERROMPIDO ===")
            self.logger.info(f"Total de tentativas: {total_attempts}")
            self.logger.info(f"Envios bem-sucedidos: {successful_sends}")
            self.logger.info(f"Taxa de sucesso final: {success_rate:.1f}%")
        except Exception as e:
            self.logger.error(f"Erro no monitoramento: {str(e)}")
        finally:
            # Liberar câmera
            if hasattr(self, 'cap') and self.cap is not None:
                self.cap.release()
                self.logger.info("✓ Câmera liberada")
    
    def analyze_image_from_path(self, image_path_or_url, c=None, show_results=True, save_result=False, output_dir="./results"):
        """
        Analisa uma imagem de arquivo local ou URL e plota os resultados
        
        Args:
            image_path_or_url (str): Caminho local da imagem ou URL
            show_results (bool): Se deve mostrar os resultados na tela
            save_result (bool): Se deve salvar a imagem processada
            output_dir (str): Diretório para salvar resultados
            
        Returns:
            dict: Resultados da análise com detecções e metadados
        """
        try:
            self.logger.info(f"Fonte: {image_path_or_url}")
            
            # Carregar imagem (local ou URL)
            original_image = self._load_image_from_source(image_path_or_url)
            if original_image is None:
                return None
            
            # Converter BGR para RGB para manter cores corretas
            original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
            
            analysis_timestamp = datetime.now()
            self.logger.info(f"Imagem: {original_image.shape[1]}x{original_image.shape[0]}")
            
            # Carregar modelo
            if c is None: c = self.CONFIDENCE_THRESHOLD

            model = YOLO(self.MODEL_PATH, verbose=False)
            
            # Executar detecção
            results = model(original_image_rgb, conf=c, verbose=False)
            
            # Processar resultados
            detections = []
            for result in results:
                for box in result.boxes:
                    detections.append({
                        'class': result.names[int(box.cls)],
                        'confidence': round(float(box.conf), 6),
                        'bbox': [round(coord, 2) for coord in box.xyxy[0].tolist()],
                        'bbox_normalized': [round(coord, 6) for coord in box.xywhn[0].tolist()]
                    })
            
            self.logger.info(f"✓ Detecções encontradas: {len(detections)}")
            
            # Criar metadados detalhados
            metadata = {
                'source': image_path_or_url,
                'analysis_timestamp': analysis_timestamp.isoformat(),
                'model_info': {
                    'model_path': self.MODEL_PATH,
                    'confidence_threshold': self.CONFIDENCE_THRESHOLD
                },
                'image_info': {
                    'original_shape': original_image.shape,
                    'channels': original_image.shape[2] if len(original_image.shape) > 2 else 1
                },
                'inference_speed': {
                    'preprocess_ms': round(results[0].speed['preprocess'], 4),
                    'inference_ms': round(results[0].speed['inference'], 4),
                    'postprocess_ms': round(results[0].speed['postprocess'], 4)
                },
                'detection_summary': {
                    'total_detections': len(detections),
                    'max_confidence': max([d['confidence'] for d in detections]) if detections else 0,
                    'classes_detected': list(set([d['class'] for d in detections])) if detections else []
                }
            }

            # Plot results
            if show_results:
                import matplotlib.pyplot as plt
                import matplotlib.patches as patches
                
                fig, ax = plt.subplots(figsize=(8, 8), facecolor='#202124')
                ax.imshow(original_image_rgb)  # Usar a imagem convertida para RGB
                h, w = original_image.shape[:2]
                
                # Add titles
                src = image_path_or_url.split('/')[-1]
                plt.suptitle(src, color='#DEDEDE', fontsize=14, fontweight='bold', y=0.98)
                plt.title(f"{len(detections)} Detecções | Limiar: {self.CONFIDENCE_THRESHOLD} | Classe: {detections[0]['class'] if detections else 'N/A'}",
                        color='white', fontsize=10, pad=10)

                # Draw detections
                for d in detections:
                    x1, y1, x2, y2 = [max(0, min(c, w if i%2 else h)) for i, c in enumerate(d['bbox'])]
                    rect = patches.Rectangle((x1, y1), x2-x1, y2-y1, linewidth=2, edgecolor='white', facecolor='none', alpha=0.8)
                    ax.add_patch(rect)
                    ax.text(x1, max(10, y1-10), f"{d['confidence']:.2f}", 
                            color='black', backgroundcolor='white', fontsize=9, fontweight='bold', alpha=0.9,
                            bbox=dict(facecolor='white', edgecolor='none', pad=1))

                # Finalize plot
                ax.set_xlim(0, w)
                ax.set_ylim(h, 0)
                plt.axis('off')
                plt.tight_layout()
                plt.show()
            
            # Retornar resultados estruturados
            return {
                'success': True,
                'detections': detections,
                'metadata': metadata,
                'original_image': original_image_rgb,  # Retornar imagem em RGB
            }
            
        except Exception as e:
            self.logger.error(f"✗ Erro na análise da imagem: {str(e)}")
            return {
                'success': False,
                'error': str(e),
                'detections': [],
                'metadata': {}
            }
        
    def _load_image_from_source(self, source):
        """Carrega imagem de arquivo local ou URL"""
        try:
            if source.startswith(('http://', 'https://')):
                # Carregar de URL
                self.logger.debug("Baixando imagem da URL...")
                response = requests.get(source, timeout=30)
                response.raise_for_status()
                
                # Converter bytes para array numpy
                import numpy as np
                image_array = np.frombuffer(response.content, np.uint8)
                image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                
                if image is None:
                    raise ValueError("Não foi possível decodificar imagem da URL")
                    
            else:
                # Carregar de arquivo local
                self.logger.debug("Carregando imagem do arquivo local...")
                image = cv2.imread(source)
                
                if image is None:
                    raise ValueError(f"Não foi possível carregar imagem: {source}")
            
            return image
            
        except requests.exceptions.RequestException as e:
            self.logger.error(f"Erro ao baixar imagem da URL: {str(e)}")
            return None
        except Exception as e:
            self.logger.error(f"Erro ao carregar imagem: {str(e)}")
            return None
    
    def _display_analysis_results(self, original_image, processed_image, detections, source):
        """Exibe resultados da análise de imagem isolada"""
        try:
            processed_rgb = cv2.cvtColor(processed_image, cv2.COLOR_BGR2RGB)
            
            # Configurações minimalistas
            title_color = '#DEDEDE'  # Preto simples
            
            # Figura minimalista
            plt.figure(figsize=(12, 8))
            
            # Mostrar imagem sem bordas ou elementos extras
            plt.imshow(processed_rgb)
            
            # Definir título e subtítulo
            plt.title(f'{source.split("/")[-1]}', color=title_color, pad=10)
            
            # Mostrar número de detecções como subtítulo
            if detections: plt.suptitle(f"{len(detections)} manchas detectadas")
            else: plt.suptitle("Nenhuma mancha detectada")
            
            plt.axis('off')
            plt.tight_layout()
            plt.grid(False)
            plt.show()
            
        except Exception as e:
            self.logger.warning(f"Erro ao exibir resultados da análise: {str(e)}")
    
    def _save_analysis_result(self, processed_image, detections, metadata, output_dir):
        """Salva resultado da análise"""
        try:
            import os
            os.makedirs(output_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')[:-3]
            
            # Salvar imagem processada
            image_filename = f"analysis_{timestamp}_{len(detections)}det.png"
            image_path = os.path.join(output_dir, image_filename)
            cv2.imwrite(image_path, processed_image)
            
            # Salvar metadados JSON
            json_filename = f"analysis_{timestamp}_metadata.json"
            json_path = os.path.join(output_dir, json_filename)
            with open(json_path, 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2, ensure_ascii=False)
            
            self.logger.info(f"✓ Resultados salvos em: {output_dir}")
            self.logger.info(f"  - Imagem: {image_filename}")
            self.logger.info(f"  - Metadados: {json_filename}")
            
        except Exception as e:
            self.logger.error(f"Erro ao salvar resultados: {str(e)}")
    
    def __del__(self):
        """Destructor method to ensure proper camera resource release"""
        if hasattr(self, 'cap') and self.cap is not None:
            self.cap.release()