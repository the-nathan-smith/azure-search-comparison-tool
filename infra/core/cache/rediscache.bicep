param redisCacheName string
param location string
param tags object = {}
param redisSkuName string = 'Standard'
param redisSkuCapacity int = 1
param resourceToken string

resource redisCache 'Microsoft.Cache/Redis@2023-08-01' = {
  name: !empty(redisCacheName) ? redisCacheName : 'redis-${resourceToken}'
  location: location
  tags: tags
  sku: {
    name: redisSkuName
    capacity: redisSkuCapacity
  }
  properties: {
    enableNonSslPort: false
    minimumTlsVersion: '1.2'
    redisConfiguration: {
      maxmemory-policy: 'allkeys-lru'
    }
  }
}
