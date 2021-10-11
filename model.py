import jax
import flax.linen as nn
import jax.numpy as jnp


class CAMlp(nn.Module):
    """
    Mlp in Channel Attention Module.
    """ 
    
    @nn.compact
    def __call__(self, inputs):
        
        x = nn.Dense(
            features=inputs.shape[-1]//2,
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False)(
                inputs)
        x = nn.relu(x)
        output = nn.Dense(
            features=inputs.shape[-1],
            dtype=jnp.float32,
            kernel_init=nn.initializers.xavier_uniform(),
            use_bias=False)(
                x)
        
        return output
    
    
class ChannelAttention(nn.Module):
    """
    Channel Attention Module
    """
    
    def setup(self):
        
        self.mlp = CAMlp()
    
    @nn.compact
    def __call__(self, x):
        avg_out = nn.avg_pool(x, window_shape=(64,64))
        avg_out = self.mlp(avg_out)
        max_out = nn.max_pool(x, window_shape=(64,64))
        max_out = self.mlp(max_out)
        
        out = avg_out + max_out
        
        return nn.sigmoid(out)
    
    
class SpatialAttention(nn.Module):
    """
    Spatial Attention Module
    """
    
    @nn.compact
    def __call__(self,x):
        avg_out = jnp.mean(x, axis=-1, keepdims=True)
        max_out = jnp.max(x, axis=-1, keepdims=True)
        
        x = jnp.concatenate((avg_out, max_out), axis=-1)
        x = nn.Conv(features=1,kernel_size=(3,3),use_bias=False)(x)
        
        return nn.sigmoid(x)   
    
    
class CBAMBlock(nn.Module):
    """
    CBAM Block
    """
    
    def setup(self):
        
        self.ca = ChannelAttention()
        self.sa = SpatialAttention()
        
    @nn.compact
    def __call__(self, x):
        
        x = self.ca(x) * x
        x = self.sa(x) * x
        
        return x
        
        
class CBAMResBlock(nn.Module):
    
    """
    CBAM integrated with a ResBlock in ResNet
    """
  
    def setup(self):
        self.conv1 = nn.Conv(64, (3,3), use_bias=False)
        self.bn1 = nn.BatchNorm(use_running_average=True)
        self.relu = nn.relu
        self.conv2 = nn.Conv(64, (3,3), use_bias=False)
        self.bn2 = nn.BatchNorm(use_running_average=True)
        
        self.cbam = CBAMBlock()
        
    @nn.compact    
    def __call__(self, x):
        
        residual = x
        
        out = nn.Conv(64, (3,3), use_bias=False)(x)
        out = nn.BatchNorm()(out)
        out = nn.gelu(out)
        
        out = nn.Conv(64, (3,3), use_bias=False)(out)
        out = nn.BatchNorm()(out)
        
        out = self.cbam(out)
        
        if out.shape != residual.shape:
            residual = nn.Conv(out.shape[-1], (1,1),
                               name='conv_projection')(residual)
        
        out += residual
        out = nn.gelu(out)
        
        return out
