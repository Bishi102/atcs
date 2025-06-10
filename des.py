"""
Discrete Event Simulator

- Simualtes the average daily profit of a coffee shop

Parameters are:
- coffeePrice: price of a cup of coffee
    - 
- numBaristas: number of baristas
- wagePerHour: hourly wage of an employee
- shopStartTime: 24h time of shop opening time
- shopOpenHours: 24h time of shop open hours
"""

import heapq
import random
import math

class Event:
    def __init__(self, time, eventType):
        self.time = time
        self.eventType = eventType

    def __lt__(self, other):
        return self.time < other.time


class CoffeeShopSimulator:
    def __init__(self,
                 numBaristas = 2,
                 coffeePrice = 5.0,
                 wagePerHour = 20.0,
                 shopStartTime = 6.0,
                 shopOpenHours = 12,
                 maxQueueLength = 5):
        # possible parameters
        self.numBaristas = numBaristas
        self.coffeePrice = coffeePrice
        self.wagePerHour = wagePerHour
        self.startTime = shopStartTime
        self.shopOpenHours = shopOpenHours
        self.maxQueueLength = maxQueueLength
        # dependent parameters
        self.customerPenalty = coffeePrice * 0.3
        self.serviceTimeMean = self._computeServiceTimeMean()
        self.basePrice = 2.0
        self.priceElasticity = -0.2

        self.busyBaristas = 0
        self.queue = []

    def _computeServiceTimeMean(self):
        minTime = 1.5 / 60
        maxTime = 5.0 / 60
        baseWage = 20.0
        k = 0.4
        wageDiff = max(self.wagePerHour - baseWage, 0)
        return minTime + (maxTime - minTime) * math.exp(-k * wageDiff)

    def _gaussian(self, x, mu, sigma, scale):
        return scale * math.exp(-((x - mu) ** 2) / (2 * sigma ** 2))

    def _timeDependentRate(self, tHour):
        baseRate = self._gaussian(tHour, mu=7, sigma=0.7, scale=350) + \
                   self._gaussian(tHour, mu=9, sigma=0.7, scale=330)
        priceFactor = (self.coffeePrice / self.basePrice) ** self.priceElasticity
        return baseRate * priceFactor

    def _getNextArrival(self, currentTime, stepSize=1/60):
        t = currentTime + stepSize
        endTime = self.startTime + self.shopOpenHours
        while t <= endTime:
            rate = self._timeDependentRate(t)
            prob = rate * stepSize
            if random.random() < prob:
                return t
            t += stepSize
        return float('inf')

    def _getServiceTime(self):
        return random.expovariate(1 / self.serviceTimeMean)

    def run(self):
        # results
        totalProfit = 0
        totalRevenue = 0
        totalWages = 0
        totalPenalties = 0

        for _ in range(365):
            # resetting daily state
            self.queue = []
            self.busyBaristas = 0
            self.totalRevenue = 0
            self.totalPenalty = 0

            currentTime = self.startTime
            endTime = self.startTime + self.shopOpenHours
            eventQueue = []
            heapq.heappush(eventQueue, Event(self._getNextArrival(currentTime), 'arrival'))

            while eventQueue:
                event = heapq.heappop(eventQueue)
                currentTime = event.time
                if currentTime > endTime:
                    break

                if event.eventType == 'arrival':
                    nextArrival = self._getNextArrival(currentTime)
                    if nextArrival <= endTime:
                        heapq.heappush(eventQueue, Event(nextArrival, 'arrival'))

                    if self.busyBaristas < self.numBaristas:
                        self.busyBaristas += 1
                        serviceTime = self._getServiceTime()
                        heapq.heappush(eventQueue, Event(currentTime + serviceTime, 'departure'))
                        self.totalRevenue += self.coffeePrice
                    elif len(self.queue) < self.maxQueueLength:
                        self.queue.append(currentTime)
                    else:
                        self.totalPenalty += self.customerPenalty

                elif event.eventType == 'departure':
                    if self.queue:
                        self.queue.pop(0)
                        serviceTime = self._getServiceTime()
                        heapq.heappush(eventQueue, Event(currentTime + serviceTime, 'departure'))
                        self.totalRevenue += self.coffeePrice
                    else:
                        self.busyBaristas -= 1
            # daily stats
            dayWages = self.numBaristas * self.wagePerHour * self.shopOpenHours
            dayProfit = self.totalRevenue - dayWages - self.totalPenalty
            # accumulative year stats
            totalProfit += dayProfit
            totalRevenue += self.totalRevenue
            totalWages += dayWages
            totalPenalties += self.totalPenalty

        return {
            'profit': totalProfit / 365,
            'revenue': totalRevenue / 365,
            'wages': totalWages / 365,
            'penalties': totalPenalties / 365
        }
