import { PromptEditor } from '@/components/prompt-editor';
import { Form, Input, InputNumber, Switch, Select, FormInstance, Modal, Button } from 'antd';
import { useTranslation } from 'react-i18next';
import { IOperatorForm } from '../../interface';
import { SettingOutlined } from '@ant-design/icons';
import { useState, useEffect } from 'react';

// 预设服务器配置
const PRESET_SERVERS = {
  "tavily-mcp": {
    command: "npx",
    args: ["-y", "tavily-mcp@0.1.2"],
    env: {
      TAVILY_API_KEY: "your-api-key-here"
    }
  },
  "amap-maps": {
    command: "npx",
    args: [
          "-y",
          "@amap/amap-maps-mcp-server"
      ],
    env: {
        AMAP_MAPS_API_KEY: "your api key71b4a42ec170187283e2fc814"
      }
  },
  "custom": {
    env: {}  // 确保初始值为空对象
  }
};

const ServerConfigForm = ({ form }: { form: FormInstance }) => {
  const serverType = form.getFieldValue('server_config')?.url ? 'url' : 'command';
  const envValue = Form.useWatch(['server_config', 'env'], form) || {};
  const [envString, setEnvString] = useState(JSON.stringify(envValue, null, 2));
  
  const handleEnvChange = (value: string) => {
    setEnvString(value);
    try {
      const parsed = JSON.parse(value || '{}');
      form.setFieldsValue({ 
        server_config: {
          ...form.getFieldValue('server_config'),
          env: parsed
        }
      });
    } catch (error) {
      // 保持原有值避免数据丢失
    }
  };

  return (
    <>
      {serverType === 'command' && (
        <>
          <Form.Item
            name={['server_config', 'command']}
            label="执行命令"
            rules={[{ 
              required: true,
              message: '请输入执行命令（如npx）'
            }]}
          >
            <Input placeholder="例：npx" />
          </Form.Item>
          <Form.Item
            name={['server_config', 'args']}
            label="参数"
            rules={[{
              validator: (_, value) => 
                !value || Array.isArray(value) 
                  ? Promise.resolve() 
                  : Promise.reject('参数必须是数组')
            }]}
          >
            <Select mode="tags" placeholder="输入参数，按回车确认" />
          </Form.Item>
        </>
      )}

      {serverType === 'url' && (
        <>
          <Form.Item
            name={['server_config', 'url']}
            label="SSE服务地址"
            rules={[
              { required: true },
              { 
                pattern: /^https?:\/\//, 
                message: '必须以http://或https://开头' 
              }
            ]}
          >
            <Input placeholder="例：https://api.example.com/sse" />
          </Form.Item>

          <Form.Item
            name={['server_config', 'headers']}
            label="请求头"
            tooltip="JSON格式的HTTP头"
            rules={[{
              validator: (_, value) => {
                try {
                  JSON.parse(value || '{}');
                  return Promise.resolve();
                } catch {
                  return Promise.reject('必须是有效的JSON格式');
                }
              }
            }]}
          >
            <Input.TextArea 
              placeholder='{"Content-Type": "application/json"}' 
              autoSize={{ minRows: 2 }}
            />
          </Form.Item>

          <Form.Item
            name={['server_config', 'timeout']}
            label="连接超时（秒）"
            initialValue={5}
            rules={[{ 
              type: 'number', 
              min: 1,
              max: 60,
              message: '请输入1-60之间的数字' 
            }]}
          >
            <InputNumber step={1} />
          </Form.Item>

          <Form.Item
            name={['server_config', 'sse_read_timeout']}
            label="读取超时（秒）"
            initialValue={300}
            rules={[{ 
              type: 'number', 
              min: 10,
              max: 3600,
              message: '请输入10-3600之间的数字' 
            }]}
          >
            <InputNumber step={10} />
          </Form.Item>
        </>
      )}

      {serverType === 'command' && (
        <Form.Item
          name={['server_config', 'env']}
          label="环境变量"
          tooltip="JSON格式的环境变量配置"
          rules={[{
            validator: (_, value) => {
              try {
                JSON.stringify(value || {});
                return Promise.resolve();
              } catch (e) {
                return Promise.reject('必须是有效的JSON格式');
              }
            }
          }]}
        >
          <Input.TextArea
            value={envString}
            onChange={(e) => handleEnvChange(e.target.value)}
            placeholder='{"KEY": "VALUE"}'
            autoSize={{ minRows: 3 }}
          />
        </Form.Item>
      )}
    </>
  );
};

const ServerConfigModal = ({ form, visible, onClose }: { 
  form: FormInstance,
  visible: boolean,
  onClose: () => void 
}) => {
  return (
    <Modal
      title="服务器高级配置"
      open={visible}
      onCancel={onClose}
      footer={null}
      width={800}
    >
      <ServerConfigForm form={form} />
    </Modal>
  );
};

const McpForm = ({ onValuesChange, form }: IOperatorForm) => {
  const { t } = useTranslation('flow');
  const [showServerConfig, setShowServerConfig] = useState(false);

  // 修改后的 useEffect
  useEffect(() => {
    const initialPreset = 'tavily-mcp';
    const selected = PRESET_SERVERS[initialPreset];
    
    form?.setFieldsValue({ 
      serverPreset: initialPreset,
      server_config: {
        ...selected,
        env: selected.env || {}
      }
    });
  }, [form]);

  // 修改后的 handleValuesChange
  const handleValuesChange = (changedValues: any, allValues: any) => {
    console.group('值变更追踪');
    console.log('变更字段:', Object.keys(changedValues));
    console.log('变更详情:', changedValues);
    
    // 始终保留现有 server_config
    const currentServerConfig = form?.getFieldValue('server_config') || {};
    const mergedValues = { 
      ...allValues,
      server_config: currentServerConfig 
    };
    
    console.log('合并后的完整值:', mergedValues);
    console.groupEnd();

    // 处理预设变更
    if (changedValues.serverPreset) {
      const selected = PRESET_SERVERS[changedValues.serverPreset as keyof typeof PRESET_SERVERS];
      console.log('应用新预设:', selected);

      // 清除可能冲突的字段（当切换模式时）
      const cleanConfig = {
        ...selected,
        env: selected.env || {},
        // 明确删除不需要的字段
        ...(selected.command ? { 
          url: undefined,
          headers: undefined,
          timeout: undefined,
          sse_read_timeout: undefined 
        } : {})
      };

      // 强制更新表单并触发值变更
      form?.setFieldsValue({
        server_config: cleanConfig
      });
      
      // 手动触发值变更回调
      onValuesChange?.({ server_config: cleanConfig }, { 
        ...allValues,
        server_config: cleanConfig 
      });
      return;
    }

    // 统一处理其他字段变更
    onValuesChange?.(changedValues, mergedValues);
  };

  return (
    <Form
      name="mcp-config"
      autoComplete="off"
      form={form}
      onValuesChange={handleValuesChange}
      layout="vertical"
      initialValues={{
        cite: false,
        prompt: "你是一个有用的助理",
        history_window: 5,
        serverPreset: 'tavily-mcp',
        server_config: { 
          ...PRESET_SERVERS['tavily-mcp']
        }
      }}
    >
      {/* 系统提示 */}
      <Form.Item
        name="prompt"
        label={t('systemPrompt')}
        tooltip={t('promptTip', { keyPrefix: 'knowledgeConfiguration' })}
        rules={[{ required: true, message: t('promptMessage') }]}
      >
        <PromptEditor 
          value={form?.getFieldValue('prompt')} 
          onChange={(value) => form?.setFieldValue('prompt', value)} 
        />
      </Form.Item>

      {/* 服务器配置选择器 */}
      <Form.Item
        name="serverPreset"
        label="服务器配置方案"
        tooltip="选择预配置方案或自定义配置"
        style={{ display: 'inline-block', width: 'calc(100% - 48px)' }}
      >
        <Select
          options={[
            { value: 'tavily-mcp', label: 'Tavily 搜索引擎' },
            { value: 'amap-maps', label: '高德地图服务' },
            { value: 'custom', label: '自定义配置' }
          ]}
        />
      </Form.Item>
      <Button 
        type="link" 
        icon={<SettingOutlined />}
        onClick={() => setShowServerConfig(true)}
        style={{ marginLeft: 8 }}
      />

      {/* 动态服务器配置表单 */}
      <ServerConfigModal
        form={form!}
        visible={showServerConfig}
        onClose={() => setShowServerConfig(false)}
      />

      {/* 引用开关 */}
      <Form.Item
        name="cite"
        label={t('enableCitation')}
        valuePropName="checked"
        tooltip="是否在回答中显示引用来源"
      >
        <Switch />
      </Form.Item>

      {/* 历史窗口 */}
      <Form.Item
        name="history_window"
        label={t('historyWindow')}
        tooltip="保留的对话历史轮数"
      >
        <InputNumber min={0} max={20} />
      </Form.Item>
    </Form>
  );
};

export default McpForm; 