#!/usr/bin/env python3
"""
Complete MongoDB TicketHolder Optimization Comparison
Includes all optimization strategies in a single file.
"""

import argparse
import numpy as np
import simpy
import matplotlib.pyplot as plt
from collections import defaultdict, deque
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Any, Optional, Callable
from enum import Enum
import random
import time


class Priority(Enum):
    """MongoDB admission priority levels."""
    kExempt = 0  # Administrative/internal operations
    kNormal = 1  # Regular client operations


class ResizePolicy(Enum):
    """Ticket resize policies."""
    kGradual = "gradual"
    kImmediate = "immediate"


class ConcurrencyAlgorithm(Enum):
    """Concurrency adjustment algorithms."""
    FIXED = "fixedConcurrentTransactions"
    THROUGHPUT_PROBING = "throughputProbing"


@dataclass
class AdmissionContext:
    """Context for admission control tracking."""
    priority: Priority = Priority.kNormal
    operation_id: int = 0
    arrival_time: float = 0.0
    operation_type: str = ""
    
    def getPriority(self) -> Priority:
        return self.priority


@dataclass
class QueueStats:
    """Comprehensive queue statistics matching MongoDB."""
    totalAddedQueue: int = 0
    totalRemovedQueue: int = 0
    totalStartedProcessing: int = 0
    totalFinishedProcessing: int = 0
    totalCanceled: int = 0
    totalTimeQueuedMicros: int = 0
    totalTimeProcessingMicros: int = 0
    totalNewAdmissions: int = 0
    totalDelinquentAcquisitions: int = 0


@dataclass
class FlowControlState:
    """Flow control state for replication lag management."""
    enabled: bool = True
    target_lag_seconds: float = 10.0
    current_lag: float = 0.0
    tokens_per_second: float = 1000000.0  # Start unlimited
    last_update: float = 0.0
    sustainer_rate: float = 1000000.0
    
    def should_throttle(self) -> bool:
        return self.enabled and self.current_lag > self.target_lag_seconds


@dataclass
class SimParams:
    """Simulation parameters matching MongoDB defaults."""
    # Ticket pools
    storageEngineConcurrentReadTransactions: int = 128
    storageEngineConcurrentWriteTransactions: int = 128
    
    # Queue depths
    storageEngineReadMaxQueueDepth: int = 1000000
    storageEngineWriteMaxQueueDepth: int = 1000000
    
    # Connection limits
    maxIncomingConnections: int = 65536
    
    # Adjustment parameters
    storageEngineConcurrencyAdjustmentAlgorithm: str = "throughputProbing"
    storageEngineConcurrencyAdjustmentIntervalMillis: int = 100
    
    # Flow control
    flowControlEnabled: bool = True
    flowControlTargetLagSeconds: float = 10.0
    
    # Workload parameters
    read_ratio: float = 0.8
    arrival_rate: float = 32.0
    burst_cv: float = 3.0
    srv_read_ms: float = 2.0
    srv_write_ms: float = 8.0
    srv_query_ms: float = 50.0
    sim_seconds: float = 60.0
    seed: int = 42
    query_ratio: float = 0.1
    yield_interval_ms: float = 10.0
    exempt_operation_ratio: float = 0.02  # 2% of operations are exempt
    
    # Delinquent operation threshold
    delinquentMs: int = 1000
    
    def __post_init__(self):
        # Shorter names for compatibility
        self.T = self.storageEngineConcurrentReadTransactions


class MongoDBEnhancedImplementation:
    """Enhanced MongoDB implementation with all three control layers."""
    
    def __init__(self, env: simpy.Environment, params: SimParams):
        self.env = env
        self.params = params
        
        # LAYER 1: Admission Control (Connection/Request Level)
        self.active_connections = 0
        self.admission_queue_depth = 0
        self.max_connections = params.maxIncomingConnections
        self.max_admission_queue = 1000  # Before AdmissionQueueOverflow
        
        # LAYER 2: Execution Control (Ticket Level) 
        self.read_tickets = params.storageEngineConcurrentReadTransactions
        self.write_tickets = params.storageEngineConcurrentWriteTransactions
        self.read_available = self.read_tickets
        self.write_available = self.write_tickets
        self.read_active = 0
        self.write_active = 0
        self.read_waiters = 0
        self.write_waiters = 0
        
        # Original ticket counts for resize operations
        self._outof_read = self.read_tickets
        self._outof_write = self.write_tickets
        
        # Execution queue limits
        self.max_read_queue = params.storageEngineReadMaxQueueDepth
        self.max_write_queue = params.storageEngineWriteMaxQueueDepth
        
        # Futex-style waiting
        self.read_futex_events = []
        self.write_futex_events = []
        
        # LAYER 3: Flow Control
        self.flow_control = FlowControlState(
            enabled=params.flowControlEnabled,
            target_lag_seconds=params.flowControlTargetLagSeconds
        )
        
        # Statistics tracking
        self.admission_stats = self._create_admission_stats()
        self.read_queue_stats = QueueStats()
        self.write_queue_stats = QueueStats()
        self.exempt_queue_stats = QueueStats()
        
        # Throughput probing state
        self.throughput_history = deque(maxlen=10)
        self.last_adjustment_time = 0
        self.last_throughput = 0
        self.operations_since_adjustment = 0
        
        # Resize policy
        self.resize_policy = ResizePolicy.kGradual
        
        # Callbacks
        self.delinquent_op_callback: Optional[Callable] = None
        
        print(f"MongoDB Enhanced: {self.read_tickets} read + {self.write_tickets} write tickets")
        print(f"  Concurrency Algorithm: {params.storageEngineConcurrencyAdjustmentAlgorithm}")
        print(f"  Flow Control: {'Enabled' if params.flowControlEnabled else 'Disabled'}")
    
    def _create_admission_stats(self):
        return {
            'total_requests': 0,
            'admitted_requests': 0,
            'rejected_requests': 0,
            'total_admission_time': 0,
            'exemptOperations': 0
        }
    
    def _tryAcquireNormalPriorityTicket(self, admCtx: AdmissionContext, is_read: bool):
        """Try to acquire a ticket with atomic compare-and-swap."""
        if is_read:
            available = self.read_available
            while True:
                if available <= 0:
                    return None
                # Simulate atomic compare-and-swap
                if self.read_available == available:
                    self.read_available -= 1
                    self.read_active += 1
                    self.read_queue_stats.totalStartedProcessing += 1
                    return time.time()  # Return acquisition time
                available = self.read_available
        else:
            available = self.write_available
            while True:
                if available <= 0:
                    return None
                if self.write_available == available:
                    self.write_available -= 1
                    self.write_active += 1
                    self.write_queue_stats.totalStartedProcessing += 1
                    return time.time()
                available = self.write_available
    
    def _performWaitForTicketUntil(self, admCtx: AdmissionContext, is_read: bool, deadline: float):
        """Wait for ticket with jitter and queue management."""
        hasStartedWaiting = False
        
        while self.env.now < deadline:
            # Quick path - try immediate acquisition
            acquisition_time = self._tryAcquireNormalPriorityTicket(admCtx, is_read)
            if acquisition_time is not None:
                return acquisition_time
            
            # Slow path - queue management
            if not hasStartedWaiting:
                if is_read:
                    previousWaiterCount = self.read_waiters
                    self.read_waiters += 1
                    max_queue = self.max_read_queue
                    self.read_queue_stats.totalAddedQueue += 1
                else:
                    previousWaiterCount = self.write_waiters
                    self.write_waiters += 1
                    max_queue = self.max_write_queue
                    self.write_queue_stats.totalAddedQueue += 1
                
                hasStartedWaiting = True
                
                # Check queue overflow
                if previousWaiterCount >= max_queue:
                    raise RuntimeError("MongoDB is overloaded and cannot accept new operations. Try again later.")
            
            # Calculate next deadline with jitter
            baseIntervalMs = 500
            jitterFactor = 0.2
            offset = np.random.uniform(-jitterFactor * baseIntervalMs, jitterFactor * baseIntervalMs)
            nextDeadline = min(deadline, self.env.now + (baseIntervalMs + offset) / 1000)
            
            # Create futex event
            wait_event = self.env.event()
            wait_start = self.env.now
            
            if is_read:
                self.read_futex_events.append((wait_event, wait_start))
            else:
                self.write_futex_events.append((wait_event, wait_start))
            
            # Wait until next deadline
            try:
                yield wait_event | self.env.timeout(nextDeadline - self.env.now)
                
                if wait_event.triggered:
                    # We got a ticket!
                    return time.time()
            except simpy.Interrupt:
                # Timeout or cancellation
                if is_read:
                    self.read_queue_stats.totalCanceled += 1
                else:
                    self.write_queue_stats.totalCanceled += 1
                raise
        
        # Deadline exceeded
        return None
    
    def _releaseNormalPriorityTicket(self, admCtx: AdmissionContext, is_read: bool, acquisition_time: float):
        """Release ticket with intelligent notification."""
        # Update processing time statistics
        processing_time_micros = int((time.time() - acquisition_time) * 1_000_000)
        
        if is_read:
            self.read_active -= 1
            self.read_queue_stats.totalFinishedProcessing += 1
            self.read_queue_stats.totalTimeProcessingMicros += processing_time_micros
            
            # Check for delinquent operation
            if processing_time_micros > self.params.delinquentMs * 1000:
                self.read_queue_stats.totalDelinquentAcquisitions += 1
                if self.delinquent_op_callback:
                    self.delinquent_op_callback(admCtx, processing_time_micros // 1000)
            
            # Intelligent notification - only notify if transitioning from 0 to 1
            availableBeforeIncrementing = self.read_available
            self.read_available += 1
            
            if availableBeforeIncrementing == 0 and self.read_waiters > 0 and self.read_futex_events:
                # Wake one random waiter
                event_idx = np.random.randint(0, len(self.read_futex_events))
                event, wait_start = self.read_futex_events.pop(event_idx)
                
                # Update queue statistics
                queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
                self.read_queue_stats.totalTimeQueuedMicros += queue_time_micros
                self.read_queue_stats.totalRemovedQueue += 1
                
                self.read_waiters -= 1
                self.read_available -= 1
                self.read_active += 1
                self.read_queue_stats.totalStartedProcessing += 1
                event.succeed()
        else:
            # Similar logic for writes
            self.write_active -= 1
            self.write_queue_stats.totalFinishedProcessing += 1
            self.write_queue_stats.totalTimeProcessingMicros += processing_time_micros
            
            if processing_time_micros > self.params.delinquentMs * 1000:
                self.write_queue_stats.totalDelinquentAcquisitions += 1
                if self.delinquent_op_callback:
                    self.delinquent_op_callback(admCtx, processing_time_micros // 1000)
            
            availableBeforeIncrementing = self.write_available
            self.write_available += 1
            
            if availableBeforeIncrementing == 0 and self.write_waiters > 0 and self.write_futex_events:
                event_idx = np.random.randint(0, len(self.write_futex_events))
                event, wait_start = self.write_futex_events.pop(event_idx)
                
                queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
                self.write_queue_stats.totalTimeQueuedMicros += queue_time_micros
                self.write_queue_stats.totalRemovedQueue += 1
                
                self.write_waiters -= 1
                self.write_available -= 1
                self.write_active += 1
                self.write_queue_stats.totalStartedProcessing += 1
                event.succeed()
    
    def _admission_control(self, admCtx: AdmissionContext):
        """MongoDB's admission control layer."""
        # Priority check - exempt operations skip queuing
        if admCtx.getPriority() == Priority.kExempt:
            self.admission_stats['exemptOperations'] += 1
            self.exempt_queue_stats.totalStartedProcessing += 1
            return True
        
        # Connection limit check
        if self.active_connections >= self.max_connections:
            raise RuntimeError("Connection limit exceeded")
        
        # Admission queue overflow check
        if self.admission_queue_depth >= self.max_admission_queue:
            self.admission_stats['rejected_requests'] += 1
            raise RuntimeError("MongoDB is overloaded and cannot accept new operations. Try again later.")
        
        # Flow control check (Layer 3)
        if self.flow_control.should_throttle():
            # Apply flow control throttling
            tokens_needed = 1
            current_tokens = self.flow_control.tokens_per_second * (self.env.now - self.flow_control.last_update)
            if current_tokens < tokens_needed:
                # Must wait for flow control
                wait_time = (tokens_needed - current_tokens) / self.flow_control.tokens_per_second
                yield self.env.timeout(wait_time)
            self.flow_control.last_update = self.env.now
        
        # Admitted successfully
        self.admission_stats['total_requests'] += 1
        self.admission_stats['admitted_requests'] += 1
        self.active_connections += 1
        return True
    
    def _throughput_probing_adjustment(self):
        """Implement MongoDB's throughput probing algorithm."""
        if self.params.storageEngineConcurrencyAdjustmentAlgorithm != "throughputProbing":
            return
        
        # Check if it's time to adjust
        if (self.env.now - self.last_adjustment_time) < (self.params.storageEngineConcurrencyAdjustmentIntervalMillis / 1000):
            return
        
        # Calculate current throughput
        current_throughput = self.operations_since_adjustment / (self.env.now - self.last_adjustment_time)
        self.throughput_history.append(current_throughput)
        
        if len(self.throughput_history) >= 3:
            # Simple probing: increase tickets if throughput is increasing
            recent_trend = self.throughput_history[-1] - self.throughput_history[-3]
            
            if recent_trend > 0:
                # Throughput increasing, try adding more tickets
                new_read_size = min(self._outof_read + 4, 256)  # Cap at 256
                new_write_size = min(self._outof_write + 4, 256)
            elif recent_trend < -0.1 * self.throughput_history[-3]:
                # Throughput decreasing significantly, reduce tickets
                new_read_size = max(self._outof_read - 2, 64)  # Floor at 64
                new_write_size = max(self._outof_write - 2, 64)
            else:
                # Stable, no change
                new_read_size = self._outof_read
                new_write_size = self._outof_write
            
            if new_read_size != self._outof_read:
                self.resize(new_read_size, is_read=True)
            if new_write_size != self._outof_write:
                self.resize(new_write_size, is_read=False)
        
        self.last_adjustment_time = self.env.now
        self.last_throughput = current_throughput
        self.operations_since_adjustment = 0
    
    def resize(self, newSize: int, is_read: bool = True):
        """Resize ticket pools at runtime."""
        if is_read:
            current_size = self._outof_read
            difference = newSize - current_size
            
            if self.resize_policy == ResizePolicy.kGradual:
                if difference > 0:
                    # Add tickets gradually
                    for _ in range(difference):
                        self.read_available += 1
                        self._outof_read += 1
                elif difference < 0:
                    # Remove tickets gradually (wait for them to be released)
                    self._outof_read = newSize
            else:  # kImmediate
                self.read_available += difference
                self._outof_read = newSize
                self.read_tickets = newSize
        else:
            # Similar logic for write tickets
            current_size = self._outof_write
            difference = newSize - current_size
            
            if self.resize_policy == ResizePolicy.kGradual:
                if difference > 0:
                    for _ in range(difference):
                        self.write_available += 1
                        self._outof_write += 1
                elif difference < 0:
                    self._outof_write = newSize
            else:
                self.write_available += difference
                self._outof_write = newSize
                self.write_tickets = newSize
    
    def execute_point_read(self, admCtx: AdmissionContext, service_time: float):
        """Execute point read with full MongoDB lifecycle."""
        try:
            # PHASE 1: Admission Control
            yield self.env.process(self._admission_control(admCtx))
            
            # PHASE 2: Execution Control - Try ticket acquisition
            deadline = self.env.now + 300  # 5 minute timeout
            acquisition_time = yield self.env.process(
                self._performWaitForTicketUntil(admCtx, is_read=True, deadline=deadline)
            )
            
            if acquisition_time is None:
                raise RuntimeError("Timeout waiting for read ticket")
            
            # PHASE 3: Storage Engine Execution
            yield self.env.timeout(service_time)
            
            # PHASE 4: Cleanup and release
            self._releaseNormalPriorityTicket(admCtx, is_read=True, acquisition_time=acquisition_time)
            self.active_connections -= 1
            
            # Update throughput tracking
            self.operations_since_adjustment += 1
            self._throughput_probing_adjustment()
            
        except RuntimeError as e:
            self.active_connections = max(0, self.active_connections - 1)
            raise e
    
    def execute_query(self, admCtx: AdmissionContext, service_time: float, yield_interval: float):
        """Execute complex query with yielding behavior."""
        try:
            # Admission control
            yield self.env.process(self._admission_control(admCtx))
            
            # Initial ticket acquisition
            deadline = self.env.now + 300
            acquisition_time = yield self.env.process(
                self._performWaitForTicketUntil(admCtx, is_read=True, deadline=deadline)
            )
            
            if acquisition_time is None:
                raise RuntimeError("Timeout waiting for read ticket")
            
            # Execute with yielding
            remaining_time = service_time
            total_acquisition_time = acquisition_time
            
            while remaining_time > 0:
                work_time = min(yield_interval, remaining_time)
                yield self.env.timeout(work_time)
                remaining_time -= work_time
                
                # Yield if more work remains
                if remaining_time > 0:
                    self._releaseNormalPriorityTicket(admCtx, is_read=True, acquisition_time=total_acquisition_time)
                    yield self.env.timeout(0.001)  # Brief yield
                    
                    # Re-acquire ticket
                    acquisition_time = yield self.env.process(
                        self._performWaitForTicketUntil(admCtx, is_read=True, deadline=deadline)
                    )
                    if acquisition_time is None:
                        raise RuntimeError("Timeout waiting for read ticket after yield")
            
            # Final cleanup
            self._releaseNormalPriorityTicket(admCtx, is_read=True, acquisition_time=total_acquisition_time)
            self.active_connections -= 1
            
            self.operations_since_adjustment += 1
            self._throughput_probing_adjustment()
            
        except RuntimeError as e:
            self.active_connections = max(0, self.active_connections - 1)
            raise e
    
    def execute_write(self, admCtx: AdmissionContext, service_time: float):
        """Execute write operation with full lifecycle."""
        try:
            # Admission control
            yield self.env.process(self._admission_control(admCtx))
            
            # Ticket acquisition
            deadline = self.env.now + 300
            acquisition_time = yield self.env.process(
                self._performWaitForTicketUntil(admCtx, is_read=False, deadline=deadline)
            )
            
            if acquisition_time is None:
                raise RuntimeError("Timeout waiting for write ticket")
            
            # Execute
            yield self.env.timeout(service_time)
            
            # Cleanup
            self._releaseNormalPriorityTicket(admCtx, is_read=False, acquisition_time=acquisition_time)
            self.active_connections -= 1
            
            self.operations_since_adjustment += 1
            self._throughput_probing_adjustment()
            
        except RuntimeError as e:
            self.active_connections = max(0, self.active_connections - 1)
            raise e
    
    def update_flow_control_lag(self, lag_seconds: float):
        """Update replication lag for flow control."""
        self.flow_control.current_lag = lag_seconds
        
        if self.flow_control.should_throttle():
            # Calculate throttling rate based on lag
            lag_ratio = lag_seconds / self.flow_control.target_lag_seconds
            self.flow_control.tokens_per_second = self.flow_control.sustainer_rate / lag_ratio
        else:
            self.flow_control.tokens_per_second = self.flow_control.sustainer_rate
    
    def get_stats(self):
        return {
            'active': self.read_active + self.write_active,
            'capacity': self._outof_read + self._outof_write,
            'available': (self.read_available, self.write_available),
            'queue_lengths': (self.read_waiters, self.write_waiters),
            'admission_stats': self.admission_stats.copy(),
            'read_queue_stats': vars(self.read_queue_stats),
            'write_queue_stats': vars(self.write_queue_stats),
            'exempt_queue_stats': vars(self.exempt_queue_stats),
            'connections': self.active_connections,
            'flow_control': {
                'enabled': self.flow_control.enabled,
                'current_lag': self.flow_control.current_lag,
                'throttle_rate': self.flow_control.tokens_per_second
            },
            'current_ticket_counts': {
                'read': self._outof_read,
                'write': self._outof_write
            }
        }


class OptimizedSeparatePools:
    """Optimization 1: Separate TicketHolders for normal and low priority operations."""
    
    def __init__(self, env: simpy.Environment, params: SimParams):
        self.env = env
        self.params = params
        
        # Use pool_ratio from params if available
        pool_ratio = getattr(params, 'pool_ratio', 0.8)
        
        # Normal operations get pool_ratio of tickets
        normal_read_tickets = int(params.storageEngineConcurrentReadTransactions * pool_ratio)
        normal_write_tickets = int(params.storageEngineConcurrentWriteTransactions * pool_ratio)
        
        # Low priority operations get remaining tickets
        remaining_ratio = 1.0 - pool_ratio
        low_priority_tickets = int((params.storageEngineConcurrentReadTransactions + 
                                   params.storageEngineConcurrentWriteTransactions) * remaining_ratio)
        
        # Create separate ticket holders
        self.normal_holder = self._create_ticket_holder(
            "Normal", normal_read_tickets, normal_write_tickets
        )
        self.low_priority_holder = self._create_ticket_holder(
            "LowPriority", low_priority_tickets, low_priority_tickets
        )
        
        print(f"Optimized Separate Pools (ratio={pool_ratio:.1f}):")
        print(f"  Normal: {normal_read_tickets} read + {normal_write_tickets} write tickets")
        print(f"  Low Priority: {low_priority_tickets} shared tickets")
    
    def _create_ticket_holder(self, name: str, read_tickets: int, write_tickets: int):
        """Create a ticket holder with unfair scheduling policy."""
        return {
            'name': name,
            'read_tickets': read_tickets,
            'write_tickets': write_tickets,
            'read_available': read_tickets,
            'write_available': write_tickets,
            'read_active': 0,
            'write_active': 0,
            'read_waiters': 0,
            'write_waiters': 0,
            'read_futex_events': [],
            'write_futex_events': [],
            'stats': QueueStats()
        }
    
    def _try_acquire_ticket(self, holder: dict, is_read: bool):
        """Try to acquire ticket with unfair policy - running threads get preference."""
        if is_read:
            if holder['read_available'] > 0:
                holder['read_available'] -= 1
                holder['read_active'] += 1
                holder['stats'].totalStartedProcessing += 1
                return True
        else:
            if holder['write_available'] > 0:
                holder['write_available'] -= 1
                holder['write_active'] += 1
                holder['stats'].totalStartedProcessing += 1
                return True
        return False
    
    def _wait_for_ticket(self, holder: dict, is_read: bool):
        """Wait for ticket with jitter."""
        if is_read:
            holder['read_waiters'] += 1
            holder['stats'].totalAddedQueue += 1
            wait_event = self.env.event()
            wait_start = self.env.now
            holder['read_futex_events'].append((wait_event, wait_start))
        else:
            holder['write_waiters'] += 1
            holder['stats'].totalAddedQueue += 1
            wait_event = self.env.event()
            wait_start = self.env.now
            holder['write_futex_events'].append((wait_event, wait_start))
        
        # Add jitter
        jitter_ms = np.random.uniform(-100, 100)
        yield self.env.timeout(abs(jitter_ms) / 1000)
        
        return wait_event
    
    def _release_ticket(self, holder: dict, is_read: bool):
        """Release ticket - always return to pool first (unfair scheduling)."""
        if is_read:
            holder['read_active'] -= 1
            holder['stats'].totalFinishedProcessing += 1
            
            # UNFAIR: Always return to pool first
            holder['read_available'] += 1
            
            # Then wake a waiter if any exist
            if holder['read_waiters'] > 0 and holder['read_futex_events']:
                # Wake one random waiter to race for the ticket
                idx = np.random.randint(0, len(holder['read_futex_events']))
                event, wait_start = holder['read_futex_events'].pop(idx)
                
                queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
                holder['stats'].totalTimeQueuedMicros += queue_time_micros
                holder['stats'].totalRemovedQueue += 1
                holder['read_waiters'] -= 1
                
                event.succeed()
        else:
            # Similar for writes
            holder['write_active'] -= 1
            holder['stats'].totalFinishedProcessing += 1
            holder['write_available'] += 1
            
            if holder['write_waiters'] > 0 and holder['write_futex_events']:
                idx = np.random.randint(0, len(holder['write_futex_events']))
                event, wait_start = holder['write_futex_events'].pop(idx)
                
                queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
                holder['stats'].totalTimeQueuedMicros += queue_time_micros
                holder['stats'].totalRemovedQueue += 1
                holder['write_waiters'] -= 1
                
                event.succeed()
    
    def _execute_operation(self, holder: dict, admCtx: AdmissionContext, 
                         service_time: float, is_read: bool, is_query: bool = False,
                         yield_interval: float = 0.01):
        """Execute operation using specified ticket holder."""
        # Fast path - try immediate acquisition
        if self._try_acquire_ticket(holder, is_read):
            # Got ticket immediately
            pass
        else:
            # Must wait
            wait_event = yield self.env.process(self._wait_for_ticket(holder, is_read))
            yield wait_event
            
            # After waking, must race to acquire
            while not self._try_acquire_ticket(holder, is_read):
                # Lost the race, wait again
                wait_event = yield self.env.process(self._wait_for_ticket(holder, is_read))
                yield wait_event
        
        # Execute operation
        if is_query:
            # Query with yielding
            remaining_time = service_time
            while remaining_time > 0:
                work_time = min(yield_interval, remaining_time)
                yield self.env.timeout(work_time)
                remaining_time -= work_time
                
                if remaining_time > 0:
                    self._release_ticket(holder, is_read)
                    yield self.env.timeout(0.001)
                    
                    # Re-acquire
                    while not self._try_acquire_ticket(holder, is_read):
                        wait_event = yield self.env.process(self._wait_for_ticket(holder, is_read))
                        yield wait_event
        else:
            # Simple operation
            yield self.env.timeout(service_time)
        
        # Release ticket
        self._release_ticket(holder, is_read)
    
    def execute_point_read(self, admCtx: AdmissionContext, service_time: float):
        """Route to appropriate ticket holder based on priority."""
        # Note: In this design, we're using Priority.kNormal to indicate low priority operations
        # This is opposite of the intuitive naming but matches the test logic
        holder = self.low_priority_holder if admCtx.priority == Priority.kNormal else self.normal_holder
        yield self.env.process(
            self._execute_operation(holder, admCtx, service_time, is_read=True)
        )
    
    def execute_query(self, admCtx: AdmissionContext, service_time: float, yield_interval: float):
        """Route to appropriate ticket holder based on priority."""
        holder = self.low_priority_holder if admCtx.priority == Priority.kNormal else self.normal_holder
        yield self.env.process(
            self._execute_operation(holder, admCtx, service_time, is_read=True, 
                                  is_query=True, yield_interval=yield_interval)
        )
    
    def execute_write(self, admCtx: AdmissionContext, service_time: float):
        """Route to appropriate ticket holder based on priority."""
        holder = self.low_priority_holder if admCtx.priority == Priority.kNormal else self.normal_holder
        yield self.env.process(
            self._execute_operation(holder, admCtx, service_time, is_read=False)
        )
    
    def get_stats(self):
        """Aggregate statistics from both ticket holders."""
        normal = self.normal_holder
        low = self.low_priority_holder
        
        total_active = (normal['read_active'] + normal['write_active'] + 
                       low['read_active'] + low['write_active'])
        total_capacity = (normal['read_tickets'] + normal['write_tickets'] + 
                         low['read_tickets'] + low['write_tickets'])
        
        return {
            'active': total_active,
            'capacity': total_capacity,
            'normal_pool': {
                'read_active': normal['read_active'],
                'write_active': normal['write_active'],
                'read_queued': normal['read_waiters'],
                'write_queued': normal['write_waiters'],
                'read_available': normal['read_available'],
                'write_available': normal['write_available']
            },
            'low_priority_pool': {
                'read_active': low['read_active'],
                'write_active': low['write_active'],
                'read_queued': low['read_waiters'],
                'write_queued': low['write_waiters'],
                'read_available': low['read_available'],
                'write_available': low['write_available']
            },
            'queue_lengths': (
                normal['read_waiters'] + low['read_waiters'],
                normal['write_waiters'] + low['write_waiters']
            )
        }


class OptimizedSeparatePoolsWithStealing:
    """Optimization 3: Separate pools with work stealing for better utilization."""
    
    def __init__(self, env: simpy.Environment, params: SimParams):
        self.env = env
        self.params = params
        
        # Use pool_ratio from params
        pool_ratio = getattr(params, 'pool_ratio', 0.6)
        
        # Normal operations get pool_ratio of tickets
        normal_read_tickets = int(params.storageEngineConcurrentReadTransactions * pool_ratio)
        normal_write_tickets = int(params.storageEngineConcurrentWriteTransactions * pool_ratio)
        
        # Low priority operations get remaining tickets
        remaining_ratio = 1.0 - pool_ratio
        low_priority_tickets = int((params.storageEngineConcurrentReadTransactions + 
                                   params.storageEngineConcurrentWriteTransactions) * remaining_ratio)
        
        # Create separate ticket holders
        self.normal_holder = self._create_ticket_holder(
            "Normal", normal_read_tickets, normal_write_tickets
        )
        self.low_priority_holder = self._create_ticket_holder(
            "LowPriority", low_priority_tickets, low_priority_tickets
        )
        
        # Work stealing statistics
        self.steal_attempts = 0
        self.successful_steals = 0
        self.steal_denials = 0
        
        print(f"Optimized Separate Pools WITH STEALING (ratio={pool_ratio:.1f}):")
        print(f"  Normal: {normal_read_tickets} read + {normal_write_tickets} write tickets")
        print(f"  Low Priority: {low_priority_tickets} shared tickets")
        print(f"  Work stealing: ENABLED")
    
    def _create_ticket_holder(self, name: str, read_tickets: int, write_tickets: int):
        """Create a ticket holder with unfair scheduling policy."""
        return {
            'name': name,
            'read_tickets': read_tickets,
            'write_tickets': write_tickets,
            'read_available': read_tickets,
            'write_available': write_tickets,
            'read_active': 0,
            'write_active': 0,
            'read_waiters': 0,
            'write_waiters': 0,
            'read_futex_events': [],
            'write_futex_events': [],
            'stats': QueueStats(),
            'last_waiter_time': 0.0,  # Track when last waiter was seen
            'stolen_tickets': 0  # Track tickets stolen by other pool
        }
    
    def _can_steal_from(self, holder: dict, is_read: bool) -> bool:
        """Determine if we can safely steal from a holder."""
        # Conservative stealing policy:
        # 1. No waiters in the pool we're stealing from
        # 2. Pool has available tickets
        # 3. Haven't had waiters for at least 10ms
        
        if is_read:
            has_waiters = holder['read_waiters'] > 0
            has_tickets = holder['read_available'] > 0
        else:
            has_waiters = holder['write_waiters'] > 0
            has_tickets = holder['write_available'] > 0
        
        # Check time since last waiter
        time_since_waiter = self.env.now - holder['last_waiter_time']
        safe_to_steal = time_since_waiter > 0.01  # 10ms threshold
        
        return has_tickets and not has_waiters and safe_to_steal
    
    def _try_acquire_ticket(self, holder: dict, is_read: bool):
        """Try to acquire ticket from specific holder."""
        if is_read:
            if holder['read_available'] > 0:
                holder['read_available'] -= 1
                holder['read_active'] += 1
                holder['stats'].totalStartedProcessing += 1
                return True
        else:
            if holder['write_available'] > 0:
                holder['write_available'] -= 1
                holder['write_active'] += 1
                holder['stats'].totalStartedProcessing += 1
                return True
        return False
    
    def _try_acquire_with_stealing(self, primary_holder: dict, secondary_holder: dict, 
                                  is_read: bool, can_steal: bool):
        """Try to acquire ticket with optional stealing from secondary pool."""
        # First try primary pool
        if self._try_acquire_ticket(primary_holder, is_read):
            return True, primary_holder
        
        # Primary exhausted - try stealing if allowed
        if can_steal:
            self.steal_attempts += 1
            if self._can_steal_from(secondary_holder, is_read):
                if self._try_acquire_ticket(secondary_holder, is_read):
                    self.successful_steals += 1
                    secondary_holder['stolen_tickets'] += 1
                    return True, secondary_holder
            else:
                self.steal_denials += 1
        
        return False, None
    
    def _wait_for_ticket(self, holder: dict, is_read: bool):
        """Wait for ticket with jitter."""
        # Update last waiter time
        holder['last_waiter_time'] = self.env.now
        
        if is_read:
            holder['read_waiters'] += 1
            holder['stats'].totalAddedQueue += 1
            wait_event = self.env.event()
            wait_start = self.env.now
            holder['read_futex_events'].append((wait_event, wait_start))
        else:
            holder['write_waiters'] += 1
            holder['stats'].totalAddedQueue += 1
            wait_event = self.env.event()
            wait_start = self.env.now
            holder['write_futex_events'].append((wait_event, wait_start))
        
        # Add jitter
        jitter_ms = np.random.uniform(-100, 100)
        yield self.env.timeout(abs(jitter_ms) / 1000)
        
        return wait_event
    
    def _release_ticket(self, holder: dict, is_read: bool):
        """Release ticket - always return to pool first (unfair scheduling)."""
        if is_read:
            holder['read_active'] -= 1
            holder['stats'].totalFinishedProcessing += 1
            
            # UNFAIR: Always return to pool first
            holder['read_available'] += 1
            
            # Then wake a waiter if any exist
            if holder['read_waiters'] > 0 and holder['read_futex_events']:
                # Wake one random waiter to race for the ticket
                idx = np.random.randint(0, len(holder['read_futex_events']))
                event, wait_start = holder['read_futex_events'].pop(idx)
                
                queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
                holder['stats'].totalTimeQueuedMicros += queue_time_micros
                holder['stats'].totalRemovedQueue += 1
                holder['read_waiters'] -= 1
                
                event.succeed()
        else:
            # Similar for writes
            holder['write_active'] -= 1
            holder['stats'].totalFinishedProcessing += 1
            holder['write_available'] += 1
            
            if holder['write_waiters'] > 0 and holder['write_futex_events']:
                idx = np.random.randint(0, len(holder['write_futex_events']))
                event, wait_start = holder['write_futex_events'].pop(idx)
                
                queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
                holder['stats'].totalTimeQueuedMicros += queue_time_micros
                holder['stats'].totalRemovedQueue += 1
                holder['write_waiters'] -= 1
                
                event.succeed()
    
    def _execute_operation(self, primary_holder: dict, secondary_holder: dict,
                         admCtx: AdmissionContext, service_time: float, 
                         is_read: bool, can_steal: bool, is_query: bool = False,
                         yield_interval: float = 0.01):
        """Execute operation with work stealing support."""
        # Track which holder we got the ticket from
        ticket_holder = None
        
        # Fast path - try immediate acquisition with stealing
        acquired, ticket_holder = self._try_acquire_with_stealing(
            primary_holder, secondary_holder, is_read, can_steal
        )
        
        if not acquired:
            # Must wait on primary pool only
            wait_event = yield self.env.process(self._wait_for_ticket(primary_holder, is_read))
            yield wait_event
            
            # After waking, try to acquire (with stealing if we're still waiting)
            while True:
                acquired, ticket_holder = self._try_acquire_with_stealing(
                    primary_holder, secondary_holder, is_read, can_steal
                )
                if acquired:
                    break
                    
                # Lost the race, wait again
                wait_event = yield self.env.process(self._wait_for_ticket(primary_holder, is_read))
                yield wait_event
        
        # Execute operation
        if is_query:
            # Query with yielding
            remaining_time = service_time
            while remaining_time > 0:
                work_time = min(yield_interval, remaining_time)
                yield self.env.timeout(work_time)
                remaining_time -= work_time
                
                if remaining_time > 0:
                    # Release to the holder we got it from
                    self._release_ticket(ticket_holder, is_read)
                    yield self.env.timeout(0.001)
                    
                    # Re-acquire (with stealing)
                    acquired = False
                    while not acquired:
                        acquired, new_holder = self._try_acquire_with_stealing(
                            primary_holder, secondary_holder, is_read, can_steal
                        )
                        if acquired:
                            ticket_holder = new_holder
                        else:
                            wait_event = yield self.env.process(self._wait_for_ticket(primary_holder, is_read))
                            yield wait_event
        else:
            # Simple operation
            yield self.env.timeout(service_time)
        
        # Release ticket back to the holder we got it from
        self._release_ticket(ticket_holder, is_read)
    
    def execute_point_read(self, admCtx: AdmissionContext, service_time: float):
        """Execute point read with work stealing."""
        if admCtx.priority == Priority.kNormal:
            # Low priority - can steal from normal pool if needed
            yield self.env.process(
                self._execute_operation(
                    self.low_priority_holder,  # Primary
                    self.normal_holder,        # Secondary (can steal from)
                    admCtx, service_time, is_read=True, can_steal=True
                )
            )
        else:
            # High priority - never steal (maintains priority guarantee)
            yield self.env.process(
                self._execute_operation(
                    self.normal_holder,        # Primary  
                    self.low_priority_holder,  # Secondary (but can't steal)
                    admCtx, service_time, is_read=True, can_steal=False
                )
            )
    
    def execute_query(self, admCtx: AdmissionContext, service_time: float, yield_interval: float):
        """Execute query with work stealing."""
        if admCtx.priority == Priority.kNormal:
            # Low priority - can steal
            yield self.env.process(
                self._execute_operation(
                    self.low_priority_holder, self.normal_holder,
                    admCtx, service_time, is_read=True, can_steal=True,
                    is_query=True, yield_interval=yield_interval
                )
            )
        else:
            # High priority - no stealing
            yield self.env.process(
                self._execute_operation(
                    self.normal_holder, self.low_priority_holder,
                    admCtx, service_time, is_read=True, can_steal=False,
                    is_query=True, yield_interval=yield_interval
                )
            )
    
    def execute_write(self, admCtx: AdmissionContext, service_time: float):
        """Execute write with work stealing."""
        if admCtx.priority == Priority.kNormal:
            # Low priority - can steal
            yield self.env.process(
                self._execute_operation(
                    self.low_priority_holder, self.normal_holder,
                    admCtx, service_time, is_read=False, can_steal=True
                )
            )
        else:
            # High priority - no stealing
            yield self.env.process(
                self._execute_operation(
                    self.normal_holder, self.low_priority_holder,
                    admCtx, service_time, is_read=False, can_steal=False
                )
            )
    
    def get_stats(self):
        """Aggregate statistics from both ticket holders."""
        normal = self.normal_holder
        low = self.low_priority_holder
        
        total_active = (normal['read_active'] + normal['write_active'] + 
                       low['read_active'] + low['write_active'])
        total_capacity = (normal['read_tickets'] + normal['write_tickets'] + 
                         low['read_tickets'] + low['write_tickets'])
        
        steal_rate = (self.successful_steals / max(1, self.steal_attempts)) * 100 if self.steal_attempts > 0 else 0
        
        return {
            'active': total_active,
            'capacity': total_capacity,
            'normal_pool': {
                'read_active': normal['read_active'],
                'write_active': normal['write_active'],
                'read_queued': normal['read_waiters'],
                'write_queued': normal['write_waiters'],
                'read_available': normal['read_available'],
                'write_available': normal['write_available'],
                'stolen_tickets': normal['stolen_tickets']
            },
            'low_priority_pool': {
                'read_active': low['read_active'],
                'write_active': low['write_active'],
                'read_queued': low['read_waiters'],
                'write_queued': low['write_waiters'],
                'read_available': low['read_available'],
                'write_available': low['write_available'],
                'stolen_tickets': low['stolen_tickets']
            },
            'queue_lengths': (
                normal['read_waiters'] + low['read_waiters'],
                normal['write_waiters'] + low['write_waiters']
            ),
            'work_stealing': {
                'attempts': self.steal_attempts,
                'successful': self.successful_steals,
                'denied': self.steal_denials,
                'success_rate': steal_rate
            }
        }


class OptimizedHighImpactOnly:
    """Optimization 2: Limit only high-impact operations (Louis's approach)."""
    
    def __init__(self, env: simpy.Environment, params: SimParams):
        self.env = env
        self.params = params
        
        # Only high-impact operations need tickets
        self.high_impact_tickets = params.storageEngineConcurrentWriteTransactions // 2  # Conservative limit
        self.high_impact_available = self.high_impact_tickets
        self.high_impact_active = 0
        self.high_impact_waiters = 0
        self.high_impact_futex_events = []
        
        # Statistics
        self.high_impact_stats = QueueStats()
        self.fast_path_operations = 0
        self.high_impact_operations = 0
        
        print(f"Optimized High Impact Only: {self.high_impact_tickets} tickets for high-impact ops")
        print(f"  Fast operations: No tickets needed!")
    
    def _is_high_impact(self, admCtx: AdmissionContext, service_time_ms: float) -> bool:
        """Determine if operation is high-impact and needs throttling."""
        # Collection scans and index builds always high impact
        if 'scan' in admCtx.operation_type.lower() or 'index' in admCtx.operation_type.lower():
            return True
        
        # Queries are high-impact (long running)
        if admCtx.operation_type == 'query':
            return True
        
        # Writes that are expected to be slow
        if admCtx.operation_type == 'write' and service_time_ms > 20:
            return True
        
        # Fast point reads never need tickets
        if admCtx.operation_type == 'point_read' and service_time_ms < 10:
            return False
        
        # Default: fast operations don't need tickets
        return False
    
    def _try_acquire_high_impact_ticket(self):
        """Try to acquire ticket for high-impact operation."""
        if self.high_impact_available > 0:
            self.high_impact_available -= 1
            self.high_impact_active += 1
            self.high_impact_stats.totalStartedProcessing += 1
            return True
        return False
    
    def _wait_for_high_impact_ticket(self):
        """Wait for high-impact ticket with jitter."""
        self.high_impact_waiters += 1
        self.high_impact_stats.totalAddedQueue += 1
        
        wait_event = self.env.event()
        wait_start = self.env.now
        self.high_impact_futex_events.append((wait_event, wait_start))
        
        # Jitter
        jitter_ms = np.random.uniform(-100, 100)
        yield self.env.timeout(abs(jitter_ms) / 1000)
        
        return wait_event
    
    def _release_high_impact_ticket(self):
        """Release high-impact ticket with unfair scheduling."""
        self.high_impact_active -= 1
        self.high_impact_stats.totalFinishedProcessing += 1
        
        # Unfair: return to pool first
        self.high_impact_available += 1
        
        # Wake a waiter if any
        if self.high_impact_waiters > 0 and self.high_impact_futex_events:
            idx = np.random.randint(0, len(self.high_impact_futex_events))
            event, wait_start = self.high_impact_futex_events.pop(idx)
            
            queue_time_micros = int((self.env.now - wait_start) * 1_000_000)
            self.high_impact_stats.totalTimeQueuedMicros += queue_time_micros
            self.high_impact_stats.totalRemovedQueue += 1
            self.high_impact_waiters -= 1
            
            event.succeed()
    
    def _execute_fast_path(self, service_time: float):
        """Fast path - no tickets needed!"""
        self.fast_path_operations += 1
        yield self.env.timeout(service_time)
    
    def _execute_high_impact(self, service_time: float, is_query: bool = False, 
                           yield_interval: float = 0.01):
        """Execute high-impact operation with ticket control."""
        self.high_impact_operations += 1
        
        # Acquire ticket
        if not self._try_acquire_high_impact_ticket():
            wait_event = yield self.env.process(self._wait_for_high_impact_ticket())
            yield wait_event
            
            # Race to acquire after waking
            while not self._try_acquire_high_impact_ticket():
                wait_event = yield self.env.process(self._wait_for_high_impact_ticket())
                yield wait_event
        
        # Execute
        if is_query:
            # Query with yielding
            remaining_time = service_time
            while remaining_time > 0:
                work_time = min(yield_interval, remaining_time)
                yield self.env.timeout(work_time)
                remaining_time -= work_time
                
                if remaining_time > 0:
                    self._release_high_impact_ticket()
                    yield self.env.timeout(0.001)
                    
                    # Re-acquire
                    while not self._try_acquire_high_impact_ticket():
                        wait_event = yield self.env.process(self._wait_for_high_impact_ticket())
                        yield wait_event
        else:
            yield self.env.timeout(service_time)
        
        # Release
        self._release_high_impact_ticket()
    
    def execute_point_read(self, admCtx: AdmissionContext, service_time: float):
        """Execute point read - usually fast path."""
        if self._is_high_impact(admCtx, service_time * 1000):
            yield self.env.process(self._execute_high_impact(service_time))
        else:
            yield self.env.process(self._execute_fast_path(service_time))
    
    def execute_query(self, admCtx: AdmissionContext, service_time: float, yield_interval: float):
        """Execute query - always high impact."""
        yield self.env.process(
            self._execute_high_impact(service_time, is_query=True, yield_interval=yield_interval)
        )
    
    def execute_write(self, admCtx: AdmissionContext, service_time: float):
        """Execute write - check if high impact."""
        if self._is_high_impact(admCtx, service_time * 1000):
            yield self.env.process(self._execute_high_impact(service_time))
        else:
            yield self.env.process(self._execute_fast_path(service_time))
    
    def get_stats(self):
        return {
            'active': self.high_impact_active,
            'capacity': self.high_impact_tickets,
            'queue_lengths': (self.high_impact_waiters, 0),
            'high_impact_queued': self.high_impact_waiters,
            'high_impact_available': self.high_impact_available,
            'fast_path_operations': self.fast_path_operations,
            'high_impact_operations': self.high_impact_operations,
            'fast_path_percentage': (self.fast_path_operations / 
                                   max(1, self.fast_path_operations + self.high_impact_operations) * 100)
        }


def simulate_replication_lag(env: simpy.Environment, design: MongoDBEnhancedImplementation):
    """Simulate varying replication lag for flow control testing."""
    while True:
        # Simulate lag varying between 0 and 20 seconds
        lag = 10 + 10 * np.sin(env.now / 30)  # Sinusoidal pattern
        design.update_flow_control_lag(max(0, lag))
        yield env.timeout(1)  # Update every second


def generate_arrivals(env: simpy.Environment, params: SimParams, design, metrics: Dict, design_offset: int):
    """Generate request arrivals with priority support."""
    rng = np.random.default_rng(params.seed + design_offset)
    
    # Gamma distribution for inter-arrival times
    shape = 1.0 / (params.burst_cv ** 2)
    scale = params.burst_cv ** 2 / params.arrival_rate
    
    request_id = 0
    while True:
        # Generate inter-arrival time
        inter_arrival = rng.gamma(shape, scale)
        yield env.timeout(inter_arrival)
        
        # Determine operation type
        is_read = rng.random() < params.read_ratio
        
        # Determine priority
        is_exempt = rng.random() < params.exempt_operation_ratio
        priority = Priority.kExempt if is_exempt else Priority.kNormal
        
        if is_read:
            # Determine if point read or complex query
            is_query = rng.random() < params.query_ratio
            op_type = 'query' if is_query else 'point_read'
            base_time = params.srv_query_ms if is_query else params.srv_read_ms
        else:
            op_type = 'write'
            base_time = params.srv_write_ms
        
        # Generate service time (log-normal with realistic variance)
        srv_time = rng.lognormal(np.log(base_time/1000), 0.6)
        
        # Create admission context
        admCtx = AdmissionContext(
            priority=priority,
            operation_id=request_id,
            arrival_time=env.now,
            operation_type=op_type
        )
        
        # Launch operation
        request_id += 1
        env.process(execute_operation(env, design, admCtx, srv_time, metrics, request_id, params))


def execute_operation(env: simpy.Environment, design, admCtx: AdmissionContext, 
                     service_time: float, metrics: Dict, request_id: int, params: SimParams):
    """Execute a single operation and record metrics."""
    arrival_time = env.now
    
    try:
        if admCtx.operation_type == 'point_read':
            yield env.process(design.execute_point_read(admCtx, service_time))
        elif admCtx.operation_type == 'query':
            yield env.process(design.execute_query(admCtx, service_time, params.yield_interval_ms / 1000))
        else:  # write
            yield env.process(design.execute_write(admCtx, service_time))
        
        # Record completion
        completion_time = env.now
        total_latency = completion_time - arrival_time
        queue_time = total_latency - service_time
        
        op_type = admCtx.operation_type
        priority_str = 'exempt' if admCtx.priority == Priority.kExempt else 'normal'
        
        metrics['latencies'][op_type].append(total_latency * 1000)
        metrics['queue_times'][op_type].append(max(0, queue_time * 1000))
        metrics['service_times'][op_type].append(service_time * 1000)
        metrics['completions'][op_type] += 1
        metrics['priority_completions'][priority_str] += 1
        
        # Debug first few operations
        if request_id <= 3:
            print(f"    {op_type} #{request_id} ({priority_str}): queue={queue_time*1000:.1f}ms, "
                  f"service={service_time*1000:.1f}ms, total={total_latency*1000:.1f}ms")
        
    except Exception as e:
        if "overloaded" in str(e).lower():
            metrics['errors']['admission_overflow'] += 1
        else:
            metrics['errors'][admCtx.operation_type] += 1


def monitor_utilization(env: simpy.Environment, design, metrics: Dict):
    """Monitor system utilization and statistics."""
    sample_interval = 0.1
    
    while True:
        stats = design.get_stats()
        utilization = stats['active'] / stats['capacity'] if stats['capacity'] > 0 else 0
        
        metrics['utilization_samples'].append(utilization)
        metrics['queue_samples'].append(stats['queue_lengths'])
        metrics['stats_snapshots'].append({
            'time': env.now,
            'stats': stats.copy()
        })
        
        yield env.timeout(sample_interval)


def calculate_percentiles(data: List[float]) -> Tuple[float, float, float]:
    """Calculate p50, p95, p99 percentiles."""
    if not data:
        return 0.0, 0.0, 0.0
    return np.percentile(data, [50, 95, 99])


def run_sim(design_name: str, params: SimParams) -> Dict[str, Any]:
    """Run simulation for one design and return metrics."""
    env = simpy.Environment()
    
    # Initialize metrics
    metrics = {
        'latencies': defaultdict(list),
        'queue_times': defaultdict(list),
        'service_times': defaultdict(list),
        'completions': defaultdict(int),
        'priority_completions': defaultdict(int),
        'errors': defaultdict(int),
        'utilization_samples': [],
        'queue_samples': [],
        'stats_snapshots': []
    }
    
    # Create design instance
    if design_name == 'Enhanced':
        design = MongoDBEnhancedImplementation(env, params)
    elif design_name == 'SeparatePools':
        design = OptimizedSeparatePools(env, params)
    elif design_name == 'SeparatePoolsWithStealing':
        design = OptimizedSeparatePoolsWithStealing(env, params)
    elif design_name == 'HighImpactOnly':
        design = OptimizedHighImpactOnly(env, params)
    else:
        raise ValueError(f"Unknown design: {design_name}")
    
    # Use different seeds for each design
    design_offset = {
        'Enhanced': 0, 
        'SeparatePools': 1000, 
        'SeparatePoolsWithStealing': 1500,
        'HighImpactOnly': 2000
    }[design_name]
    
    # Start processes
    env.process(generate_arrivals(env, params, design, metrics, design_offset))
    env.process(monitor_utilization(env, design, metrics))
    
    # Start replication lag simulation for enhanced version
    if hasattr(design, 'update_flow_control_lag'):
        env.process(simulate_replication_lag(env, design))
    
    print(f"\nRunning {design_name}...")
    
    # Run simulation
    env.run(until=params.sim_seconds)
    
    # Calculate results
    results = {'design': design_name}
    
    total_ops = 0
    for op_type in ['point_read', 'query', 'write']:
        latencies = metrics['latencies'][op_type]
        queue_times = metrics['queue_times'][op_type]
        service_times = metrics['service_times'][op_type]
        completions = metrics['completions'][op_type]
        total_ops += completions
        
        if latencies:
            p50, p95, p99 = calculate_percentiles(latencies)
            throughput = completions / params.sim_seconds
            avg_queue = np.mean(queue_times)
            avg_service = np.mean(service_times)
        else:
            p50 = p95 = p99 = throughput = avg_queue = avg_service = 0.0
        
        results[f'{op_type}_p50'] = p50
        results[f'{op_type}_p95'] = p95
        results[f'{op_type}_p99'] = p99
        results[f'{op_type}_throughput'] = throughput
        results[f'{op_type}_avg_queue'] = avg_queue
        results[f'{op_type}_avg_service'] = avg_service
        results[f'{op_type}_latencies'] = latencies
        
        print(f"  {op_type}: {completions} ops, {throughput:.1f} ops/s, "
              f"queue={avg_queue:.1f}ms, service={avg_service:.1f}ms")
    
    # Calculate utilization
    if metrics['utilization_samples']:
        avg_utilization = np.mean(metrics['utilization_samples'])
        idle_percentage = (1.0 - avg_utilization) * 100
    else:
        avg_utilization = 0.0
        idle_percentage = 100.0
    
    results['idle_percentage'] = idle_percentage
    results['total_throughput'] = total_ops / params.sim_seconds
    results['metrics'] = metrics
    
    print(f"  Total: {total_ops} ops, {results['total_throughput']:.1f} ops/s")
    print(f"  Utilization: {avg_utilization*100:.1f}% (idle: {idle_percentage:.1f}%)")
    
    # Print priority statistics
    if 'exempt' in metrics['priority_completions'] or 'normal' in metrics['priority_completions']:
        print(f"  Exempt operations: {metrics['priority_completions']['exempt']}")
        print(f"  Normal operations: {metrics['priority_completions']['normal']}")
    
    # Print error statistics
    if metrics['errors']:
        print(f"  Errors: {dict(metrics['errors'])}")
    
    # Print design-specific statistics
    if hasattr(design, 'get_stats'):
        final_stats = design.get_stats()
        
        if 'normal_pool' in final_stats:
            # Separate pools design
            print(f"  Normal pool: {final_stats['normal_pool']['read_active']} read, "
                  f"{final_stats['normal_pool']['write_active']} write active")
            print(f"  Low priority pool: {final_stats['low_priority_pool']['read_active']} read, "
                  f"{final_stats['low_priority_pool']['write_active']} write active")
            
            # Work stealing stats if available
            if 'work_stealing' in final_stats:
                ws = final_stats['work_stealing']
                print(f"  Work stealing: {ws['successful']}/{ws['attempts']} successful "
                      f"({ws['success_rate']:.1f}% rate), {ws['denied']} denied")
                print(f"  Stolen tickets - Normal: {final_stats['normal_pool']['stolen_tickets']}, "
                      f"Low: {final_stats['low_priority_pool']['stolen_tickets']}")
        
        if 'fast_path_percentage' in final_stats:
            # High impact only design
            print(f"  Fast path operations: {final_stats['fast_path_operations']} "
                  f"({final_stats['fast_path_percentage']:.1f}%)")
            print(f"  High impact operations: {final_stats['high_impact_operations']}")
            print(f"  High impact queued: {final_stats['high_impact_queued']}")
        
        if 'current_ticket_counts' in final_stats:
            print(f"  Final ticket counts: Read={final_stats['current_ticket_counts']['read']}, "
                  f"Write={final_stats['current_ticket_counts']['write']}")
        
        if 'flow_control' in final_stats:
            print(f"  Flow control: lag={final_stats['flow_control']['current_lag']:.1f}s, "
                  f"throttle_rate={final_stats['flow_control']['throttle_rate']:.0f} ops/s")
    
    return results


def plot_optimization_comparison(results_list: List[Dict], params: SimParams):
    """Plot comparison of different optimization strategies."""
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Latency CDFs by design
    designs = []
    colors = ['blue', 'green', 'red', 'purple']
    
    for i, results in enumerate(results_list):
        design_name = results['design']
        designs.append(design_name)
        color = colors[i % len(colors)]
        
        # Combined latencies for overall CDF
        all_latencies = []
        for op_type in ['point_read', 'query', 'write']:
            all_latencies.extend(results[f'{op_type}_latencies'])
        
        if all_latencies:
            sorted_lat = np.sort(all_latencies)
            cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
            ax1.plot(sorted_lat, cdf, label=design_name, linewidth=2, color=color)
    
    ax1.set_xlabel('Latency (ms)')
    ax1.set_ylabel('CDF')
    ax1.set_title('Overall Latency Distribution by Design')
    ax1.grid(True, alpha=0.3)
    ax1.legend()
    ax1.set_xlim(left=0, right=np.percentile([r for results in results_list 
                                              for r in results['point_read_latencies'] + 
                                              results['query_latencies'] + 
                                              results['write_latencies']], 99))
    
    # 2. Throughput comparison
    op_types = ['point_read', 'query', 'write']
    x = np.arange(len(op_types))
    width = 0.2
    
    for i, results in enumerate(results_list):
        design_name = results['design']
        throughputs = [results[f'{op}_throughput'] for op in op_types]
        ax2.bar(x + i * width, throughputs, width, label=design_name, alpha=0.8)
    
    ax2.set_xlabel('Operation Type')
    ax2.set_ylabel('Throughput (ops/s)')
    ax2.set_title('Throughput by Operation Type')
    ax2.set_xticks(x + width * (len(results_list) - 1) / 2)
    ax2.set_xticklabels([op.replace('_', ' ').title() for op in op_types])
    ax2.legend()
    ax2.grid(True, alpha=0.3, axis='y')
    
    # 3. Queue time comparison
    for i, results in enumerate(results_list):
        design_name = results['design']
        color = colors[i % len(colors)]
        
        # Get queue time samples over time
        metrics = results['metrics']
        if 'stats_snapshots' in metrics and metrics['stats_snapshots']:
            times = [s['time'] for s in metrics['stats_snapshots']]
            queue_depths = [sum(s['stats']['queue_lengths']) for s in metrics['stats_snapshots']]
            ax3.plot(times, queue_depths, label=design_name, linewidth=2, color=color, alpha=0.7)
    
    ax3.set_xlabel('Time (s)')
    ax3.set_ylabel('Total Queue Depth')
    ax3.set_title('Queue Depth Over Time')
    ax3.grid(True, alpha=0.3)
    ax3.legend()
    
    # 4. Design-specific metrics
    design_names = [r['design'] for r in results_list]
    total_throughputs = [r['total_throughput'] for r in results_list]
    
    # Bar chart of total throughput with percentile overlays
    bars = ax4.bar(design_names, total_throughputs, alpha=0.6)
    
    # Add p99 latency as text on bars
    for i, (bar, results) in enumerate(zip(bars, results_list)):
        height = bar.get_height()
        p99_avg = np.mean([results[f'{op}_p99'] for op in ['point_read', 'query', 'write']])
        ax4.text(bar.get_x() + bar.get_width()/2., height,
                f'p99: {p99_avg:.1f}ms', ha='center', va='bottom')
    
    ax4.set_ylabel('Total Throughput (ops/s)')
    ax4.set_title('Total Throughput and p99 Latency by Design')
    ax4.grid(True, alpha=0.3, axis='y')
    
    # Add design-specific annotations
    for i, results in enumerate(results_list):
        if 'stats_snapshots' in results['metrics'] and results['metrics']['stats_snapshots']:
            final_stats = results['metrics']['stats_snapshots'][-1]['stats']
            if 'fast_path_percentage' in final_stats:
                # High Impact Only design
                ax4.text(i, 0, f"{final_stats['fast_path_percentage']:.0f}% fast path", 
                        ha='center', va='top', rotation=45, fontsize=8)
    
    plt.tight_layout()
    plt.suptitle(f'MongoDB TicketHolder Optimization Comparison\n'
                 f'Load: {params.arrival_rate} ops/s, '
                 f'{params.read_ratio:.0%} reads, '
                 f'{params.exempt_operation_ratio:.0%} exempt ops',
                 y=1.02)
    plt.show()


def main():
    """Main function - parse args and run simulation."""
    parser = argparse.ArgumentParser(description='Complete MongoDB TicketHolder Simulation')
    
    # MongoDB parameters
    parser.add_argument('--read-tickets', type=int, default=128,
                        help='storageEngineConcurrentReadTransactions')
    parser.add_argument('--write-tickets', type=int, default=128,
                        help='storageEngineConcurrentWriteTransactions')
    parser.add_argument('--max-connections', type=int, default=65536,
                        help='maxIncomingConnections')
    parser.add_argument('--algorithm', type=str, default='throughputProbing',
                        choices=['fixedConcurrentTransactions', 'throughputProbing'],
                        help='Concurrency adjustment algorithm')
    parser.add_argument('--adjustment-interval-ms', type=int, default=100,
                        help='Adjustment interval for throughput probing')
    
    # Flow control
    parser.add_argument('--flow-control', action='store_true', default=True,
                        help='Enable flow control')
    parser.add_argument('--flow-control-target-lag', type=float, default=10.0,
                        help='Target replication lag seconds')
    
    # Workload parameters
    parser.add_argument('--read-ratio', type=float, default=0.8, help='Read fraction')
    parser.add_argument('--arrival-rate', type=float, default=50.0, help='Arrival rate (req/s)')
    parser.add_argument('--burst-cv', type=float, default=3.0, help='Burst coefficient of variation')
    parser.add_argument('--srv-read-ms', type=float, default=2.0, help='Point read service time (ms)')
    parser.add_argument('--srv-write-ms', type=float, default=8.0, help='Write service time (ms)')
    parser.add_argument('--srv-query-ms', type=float, default=50.0, help='Query service time (ms)')
    parser.add_argument('--query-ratio', type=float, default=0.15, help='Fraction of reads that are queries')
    parser.add_argument('--exempt-ratio', type=float, default=0.02, help='Fraction of exempt operations')
    parser.add_argument('--yield-interval-ms', type=float, default=10.0, help='Query yield interval (ms)')
    parser.add_argument('--sim-seconds', type=float, default=120.0, help='Simulation duration (s)')
    parser.add_argument('--seed', type=int, default=42, help='Random seed')
    
    # Comparison mode
    parser.add_argument('--compare-designs', action='store_true', 
                        help='Compare different optimization designs')
    
    # Optimization-specific parameters
    parser.add_argument('--pool-ratio', type=float, default=0.6,
                        help='Ratio of tickets allocated to normal priority pool (vs low priority)')
    
    args = parser.parse_args()
    
    # Create parameters
    params = SimParams(
        storageEngineConcurrentReadTransactions=args.read_tickets,
        storageEngineConcurrentWriteTransactions=args.write_tickets,
        maxIncomingConnections=args.max_connections,
        storageEngineConcurrencyAdjustmentAlgorithm=args.algorithm,
        storageEngineConcurrencyAdjustmentIntervalMillis=args.adjustment_interval_ms,
        flowControlEnabled=args.flow_control,
        flowControlTargetLagSeconds=args.flow_control_target_lag,
        read_ratio=args.read_ratio,
        arrival_rate=args.arrival_rate,
        burst_cv=args.burst_cv,
        srv_read_ms=args.srv_read_ms,
        srv_write_ms=args.srv_write_ms,
        srv_query_ms=args.srv_query_ms,
        query_ratio=args.query_ratio,
        exempt_operation_ratio=args.exempt_ratio,
        yield_interval_ms=args.yield_interval_ms,
        sim_seconds=args.sim_seconds,
        seed=args.seed
    )
    
    # Add pool_ratio as an attribute if provided
    params.pool_ratio = args.pool_ratio
    
    print(f"MongoDB TicketHolder Optimization Comparison")
    print(f"  Read tickets: {params.storageEngineConcurrentReadTransactions}")
    print(f"  Write tickets: {params.storageEngineConcurrentWriteTransactions}")
    print(f"  Max connections: {params.maxIncomingConnections}")
    print(f"  Algorithm: {params.storageEngineConcurrencyAdjustmentAlgorithm}")
    print(f"  Flow control: {'Enabled' if params.flowControlEnabled else 'Disabled'}")
    print(f"  Workload: {params.arrival_rate} req/s, {params.read_ratio:.0%} reads")
    print(f"  Exempt operations: {params.exempt_operation_ratio:.1%}")
    
    if args.compare_designs:
        # Run all optimization designs
        results_list = []
        
        # 1. Current MongoDB implementation (baseline)
        results_enhanced = run_sim('Enhanced', params)
        results_list.append(results_enhanced)
        
        # 2. Separate pools for different priorities (Dani's approach)
        results_separate = run_sim('SeparatePools', params)
        results_list.append(results_separate)
        
        # 3. Separate pools with work stealing
        results_stealing = run_sim('SeparatePoolsWithStealing', params)
        results_list.append(results_stealing)
        
        # 4. Limit only high-impact operations (Louis's approach)
        results_high_impact = run_sim('HighImpactOnly', params)
        results_list.append(results_high_impact)
        
        # Print comparison table
        print(f"\n{'='*160}")
        print(f"OPTIMIZATION STRATEGY COMPARISON")
        print(f"{'='*160}")
        
        headers = ['Design', 'Total Tput', 'Read p99', 'Query p99', 'Write p99', 
                   'Avg Queue', 'Special Metrics']
        print(f"{headers[0]:<25} {headers[1]:<12} {headers[2]:<10} {headers[3]:<10} "
              f"{headers[4]:<10} {headers[5]:<10} {headers[6]:<50}")
        print("-" * 160)
        
        for results in results_list:
            design = results['design']
            total_tput = results['total_throughput']
            read_p99 = results['point_read_p99']
            query_p99 = results['query_p99']
            write_p99 = results['write_p99']
            avg_queue = np.mean([results['point_read_avg_queue'], 
                                results['query_avg_queue'], 
                                results['write_avg_queue']])
            
            # Design-specific metrics
            special = ""
            if design == 'SeparatePools':
                special = "Complete priority isolation"
            elif design == 'SeparatePoolsWithStealing':
                stats = results['metrics']['stats_snapshots'][-1]['stats']
                ws = stats['work_stealing']
                special = f"Isolation + stealing ({ws['success_rate']:.0f}% steal rate)"
            elif design == 'HighImpactOnly':
                stats = results['metrics']['stats_snapshots'][-1]['stats']
                special = f"{stats['fast_path_percentage']:.0f}% ops bypass tickets"
            elif design == 'Enhanced':
                special = "Full MongoDB implementation"
            
            print(f"{design:<25} {total_tput:<12.1f} {read_p99:<10.1f} {query_p99:<10.1f} "
                  f"{write_p99:<10.1f} {avg_queue:<10.1f} {special:<50}")
        
        print(f"\n{'='*160}")
        print("KEY INSIGHTS:")
        print(f"{'='*160}")
        
        # Calculate improvements
        baseline_tput = results_list[0]['total_throughput']
        baseline_p99 = np.mean([results_list[0]['point_read_p99'], 
                               results_list[0]['query_p99'], 
                               results_list[0]['write_p99']])
        
        for results in results_list[1:]:
            design = results['design']
            tput_improvement = ((results['total_throughput'] - baseline_tput) / baseline_tput) * 100
            p99_improvement = ((baseline_p99 - np.mean([results['point_read_p99'], 
                                                       results['query_p99'], 
                                                       results['write_p99']])) / baseline_p99) * 100
            
            print(f"{design}: {tput_improvement:+.1f}% throughput, {p99_improvement:+.1f}% p99 latency improvement")
        
        # Additional insights for work stealing
        if 'work_stealing' in results_list[2]['metrics']['stats_snapshots'][-1]['stats']:
            ws_stats = results_list[2]['metrics']['stats_snapshots'][-1]['stats']['work_stealing']
            print(f"\nWork Stealing Details: {ws_stats['successful']} successful steals out of "
                  f"{ws_stats['attempts']} attempts ({ws_stats['denied']} denied due to waiters)")
        
        # Plot comparison
        plot_optimization_comparison(results_list, params)
        
    else:
        # Run single enhanced simulation
        results = run_sim('Enhanced', params)
        
        # Simple plot for single run
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Latency CDF
        for op_type in ['point_read', 'query', 'write']:
            latencies = results[f'{op_type}_latencies']
            if latencies:
                sorted_lat = np.sort(latencies)
                cdf = np.arange(1, len(sorted_lat) + 1) / len(sorted_lat)
                ax1.plot(sorted_lat, cdf, label=op_type.replace('_', ' ').title(), linewidth=2)
        
        ax1.set_xlabel('Latency (ms)')
        ax1.set_ylabel('CDF')
        ax1.set_title('Operation Latency Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.legend()
        
        # Throughput by operation
        op_types = ['point_read', 'query', 'write']
        throughputs = [results[f'{op}_throughput'] for op in op_types]
        ax2.bar(op_types, throughputs, alpha=0.7)
        ax2.set_ylabel('Throughput (ops/s)')
        ax2.set_title('Throughput by Operation Type')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':
    main()