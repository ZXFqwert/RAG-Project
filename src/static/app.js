/**
 * RAG系统前端JavaScript
 */

// 全局变量
let currentSessionId = "default";
let isProcessing = false;

// 页面加载完成后初始化
document.addEventListener('DOMContentLoaded', function() {
    // DOM元素
    const messagesContainer = document.getElementById('messagesContainer');
    const userInput = document.getElementById('userInput');
    const sendBtn = document.getElementById('sendBtn');
    const newChatBtn = document.getElementById('newChatBtn');
    const chatHistory = document.getElementById('chatHistory');
    const esResults = document.getElementById('esResults');
    const milvusResults = document.getElementById('milvusResults');
    const esResultCount = document.getElementById('esResultCount');
    const milvusResultCount = document.getElementById('milvusResultCount');
    
    // 自动调整文本框高度
    userInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = (this.scrollHeight) + 'px';
        if (this.scrollHeight > 200) {
            this.style.overflowY = 'auto';
        } else {
            this.style.overflowY = 'hidden';
        }
    });
    
    // 发送消息
    function sendMessage() {
        if (isProcessing || !userInput.value.trim()) return;
        
        const userMessage = userInput.value.trim();
        
        // 添加用户消息到界面
        appendMessage('user', userMessage);
        
        // 清空输入框并调整高度
        userInput.value = '';
        userInput.style.height = 'auto';
        
        // 显示加载指示器
        appendMessage('assistant', '<div class="loader"></div>', true);
        
        // 清空检索结果区域
        esResults.innerHTML = '';
        milvusResults.innerHTML = '';
        esResultCount.textContent = '0';
        milvusResultCount.textContent = '0';
        
        // 设置处理标志
        isProcessing = true;
        
        // 连接WebSocket
        sendMessageViaWebSocket(userMessage);
    }
    
    // 通过WebSocket发送消息
    function sendMessageViaWebSocket(message) {
        const websocketProtocol = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${websocketProtocol}//${window.location.host}/ws/chat`;
        const ws = new WebSocket(wsUrl);
        
        let assistantResponse = '';
        let assistantMessageElement = null;
        
        ws.onopen = function() {
            ws.send(JSON.stringify({
                query: message,
                use_reflection: false,
                session_id: currentSessionId
            }));
        };
        
        ws.onmessage = function(event) {
            const response = JSON.parse(event.data);
            
            // 移除加载指示器
            if (document.querySelector('.loader')) {
                const loaderMessage = document.querySelector('.message.assistant:last-child');
                if (loaderMessage) messagesContainer.removeChild(loaderMessage);
            }
            
            switch (response.type) {
                case 'retrieval_results':
                    // 显示ES检索结果
                    if (response.es_results && response.es_results.length > 0) {
                        esResultCount.textContent = response.es_results.length;
                        response.es_results.forEach(doc => {
                            appendDocumentCard(esResults, doc);
                        });
                    }
                    
                    // 显示Milvus检索结果
                    if (response.milvus_results && response.milvus_results.length > 0) {
                        milvusResultCount.textContent = response.milvus_results.length;
                        response.milvus_results.forEach(doc => {
                            appendDocumentCard(milvusResults, doc);
                        });
                    }
                    break;
                
                case 'token':
                    // 流式显示回答
                    if (!assistantMessageElement) {
                        assistantMessageElement = appendMessage('assistant', '');
                    }
                    assistantResponse += response.token;
                    assistantMessageElement.querySelector('.content').innerHTML = assistantResponse;
                    
                    // 滚动到底部
                    messagesContainer.scrollTop = messagesContainer.scrollHeight;
                    break;
                
                case 'end':
                    // 处理完成
                    isProcessing = false;
                    
                    // 更新历史对话列表
                    updateChatHistory();
                    break;
                
                case 'error':
                    console.error('Error:', response.message);
                    appendMessage('assistant', `<span class="text-danger">发生错误: ${response.message}</span>`);
                    isProcessing = false;
                    break;
            }
        };
        
        ws.onerror = function(error) {
            console.error('WebSocket Error:', error);
            appendMessage('assistant', '<span class="text-danger">连接错误，请重试</span>');
            isProcessing = false;
        };
        
        ws.onclose = function() {
            if (isProcessing) {
                isProcessing = false;
            }
        };
    }
    
    // 添加消息到对话区域
    function appendMessage(role, content, isLoading = false) {
        const messageDiv = document.createElement('div');
        messageDiv.className = `message ${role}`;
        
        if (isLoading) {
            messageDiv.innerHTML = `
                <div class="avatar ${role}">${role === 'user' ? 'U' : 'A'}</div>
                <div class="content">${content}</div>
            `;
        } else {
            messageDiv.innerHTML = `
                <div class="avatar ${role}">${role === 'user' ? 'U' : 'A'}</div>
                <div class="content">${content}</div>
            `;
        }
        
        messagesContainer.appendChild(messageDiv);
        messagesContainer.scrollTop = messagesContainer.scrollHeight;
        
        return messageDiv;
    }
    
    // 添加文档卡片到检索结果区域
    function appendDocumentCard(container, doc) {
        const card = document.createElement('div');
        card.className = 'document-card';
        
        const metadata = doc.metadata || {};
        const source = metadata.source || '未知来源';
        const score = metadata.score ? `匹配度: ${(metadata.score * 100).toFixed(2)}%` : '';
        
        card.innerHTML = `
            <div class="document-content">${doc.content}</div>
            <div class="document-metadata">
                <div>${source}</div>
                <div>${score}</div>
            </div>
        `;
        
        container.appendChild(card);
    }
    
    // 更新历史对话列表
    function updateChatHistory() {
        fetch(`/api/chat-history?session_id=${currentSessionId}`)
            .then(response => response.json())
            .then(data => {
                const history = data.history;
                
                // 清空历史列表
                chatHistory.innerHTML = '';
                
                // 按对话分组显示
                for (let i = 0; i < history.length; i += 2) {
                    if (i + 1 < history.length) {
                        const userMessage = history[i].content;
                        
                        const historyItem = document.createElement('div');
                        historyItem.className = 'history-item';
                        historyItem.textContent = userMessage;
                        historyItem.title = userMessage;
                        
                        chatHistory.appendChild(historyItem);
                    }
                }
            })
            .catch(error => console.error('获取历史记录失败:', error));
    }
    
    // 创建新对话
    function createNewChat() {
        // 生成新的会话ID
        currentSessionId = 'session_' + new Date().getTime();
        
        // 清空消息区域，只保留欢迎消息
        messagesContainer.innerHTML = '';
        appendMessage('assistant', '你好，我是基于RAG技术的问答助手。请问有什么可以帮助你的？');
        
        // 清空检索结果
        esResults.innerHTML = '';
        milvusResults.innerHTML = '';
        esResultCount.textContent = '0';
        milvusResultCount.textContent = '0';
    }
    
    // 事件监听
    sendBtn.addEventListener('click', sendMessage);
    userInput.addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendMessage();
        }
    });
    newChatBtn.addEventListener('click', createNewChat);
    
    // 初始化
    updateChatHistory();
}); 